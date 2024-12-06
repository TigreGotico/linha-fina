import re
from typing import Tuple, Iterable, Optional, List, Dict

import joblib

try:
    import ahocorasick
except ImportError:
    ahocorasick = None


class KeywordFeatures:
    def __init__(self, csv_path: Optional[str] = None,
                 ignore_list: Optional[List[str]] = None,
                 use_automatons: Optional[bool] = None):
        """
        Initialize the KeywordFeatures class.

        Args:
            csv_path (Optional[str]): Path to the CSV file containing entities.
            ignore_list (Optional[List[str]]): List of words to ignore.
            use_automatons (bool): Whether to use Aho-Corasick automatons for matching.
        """
        if ahocorasick is None and use_automatons:
            raise ImportError("ERROR - pip install pyahocorasick")

        self.ignore_list = ignore_list or []
        self.use_automatons: bool = use_automatons or False
        self.automatons: Dict[str, ahocorasick.Automaton] = {}
        self._needs_building: List[str] = []
        self.entities: Dict[str, List[str]] = {}
        if csv_path:
            self.load_from_csv(csv_path)

    @property
    def labels(self) -> List[str]:
        """Get sorted list of entity labels."""
        return sorted(list(self.entities.keys()))

    def reset(self) -> None:
        """Reset the automatons and entities."""
        if self.use_automatons:
            self._needs_building = [name for name in self.automatons]
            self.automatons = {name: ahocorasick.Automaton() for name in self.automatons.keys()}
            for name, samples in self.entities.items():
                for s in samples:
                    self.automatons[name].add_word(s.lower(), s)
        self.entities = {}

    def register_entity(self, name: str, samples: List[str]) -> None:
        """Register runtime entity samples.

        Args:
            name (str): Name of the entity.
            samples (List[str]): List of samples for the entity.
        """
        if name not in self.entities:
            self.entities[name] = []
        self.entities[name] += samples

        if self.use_automatons:
            if name not in self.automatons:
                self.automatons[name] = ahocorasick.Automaton()
            for s in samples:
                self.automatons[name].add_word(s.lower(), s)
            self._needs_building.append(name)

    def deregister_entity(self, name: str) -> None:
        """Deregister an entity.

        Args:
            name (str): Name of the entity to deregister.
        """
        if name in self.entities:
            self.entities.pop(name)
        if name in self.automatons:
            self.automatons.pop(name)
        if name in self._needs_building:
            self._needs_building.remove(name)

    def load_from_csv(self, csv_path: str) -> Dict[str, List[str]]:
        """Load entities from a CSV file.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            Dict[str, List[str]]: Loaded entities.
        """
        ents: Dict[str, List[str]] = {}
        if isinstance(csv_path, str):
            files = [csv_path]
        else:
            files = csv_path
        data: List[Tuple[str, str]] = []
        for path in files:
            with open(path) as f:
                lines = f.read().split("\n")[1:]
                data += [tuple(l.split(",", 1)) for l in lines if "," in l]

        for n, s in data:
            if n not in ents:
                ents[n] = []
            ents[n].append(s)
            if self.use_automatons:
                self._needs_building.append(n)

        if self.use_automatons:
            for k, samples in ents.items():
                self._needs_building.append(k)
                if k not in self.automatons:
                    self.automatons[k] = ahocorasick.Automaton()
                for s in samples:
                    self.automatons[k].add_word(s.lower(), s)
        self.entities.update(ents)
        return ents

    def _voc_match(self, utt: str, entity: str) -> Iterable[str]:
        """
        Determine if the given utterance contains the vocabulary provided.

        Args:
            utt (str): Utterance to be tested.
            entity (str): Name of the vocabulary.

        Returns:
            Optional[str]: Longest match if the utterance contains the vocabulary, otherwise None.
        """
        _vocs = self.entities.get(entity, [])
        if utt and _vocs:
            for voc in _vocs:
                if len(voc) < 3:
                    continue
                if "_name" in entity and voc.lower() in self.ignore_list:
                    continue
                if re.match(r'.*\b' + re.escape(voc) + r'\b.*', utt):
                    yield voc

    def _automaton_match(self, utt: str, entity: str) -> Iterable[str]:
        """
        Determine if the given utterance contains the vocabulary provided using Aho-Corasick automaton.

        Args:
            utt (str): Utterance to be tested.
            entity (str): Name of the vocabulary.

        Returns:
            Optional[str]: Match if the utterance contains the vocabulary, otherwise None.
        """
        if entity not in self.automatons or entity not in self.entities:
            # skip automatons without registered samples
            return None

        automaton = self.automatons[entity]
        if entity in self._needs_building:
            automaton.make_automaton()
            self._needs_building.remove(entity)

        try:
            for idx, v in automaton.iter(utt):
                if len(v) < 3:
                    continue

                if "_name" in entity and v.lower() in self.ignore_list:
                    continue

                if v.lower() + " " in utt or utt.endswith(v.lower()):
                    yield v
        except AttributeError as e:
            # print("Not an Aho-Corasick automaton yet, register keywords first")
            pass
        return None

    def match(self, utt: str) -> Iterable[Tuple[str, str]]:
        """
        Match the given utterance with registered entities.

        Args:
            utt (str): Utterance to be tested.

        Returns:
            Iterable[Tuple[str, str]]: Iterable of matching entities and their values.
        """
        utt = utt.lower().strip(".!?,;:")
        for k in list(self.entities):
            matcher = self._automaton_match(utt, k) if self.use_automatons else self._voc_match(utt, k)
            for v in matcher:
                yield k, v

    def extract(self, sentence: str) -> Dict[str, str]:
        """
        Extract matching entities from the given sentence.

        Args:
            sentence (str): Sentence to be processed.

        Returns:
            Dict[str, str]: Dictionary of matching entities and their values.
        """
        match: Dict[str, str] = {}
        for k, v in self.match(sentence):
            if k not in match or len(v) > len(match[k]):
                match[k] = v
        return match

    def save(self, file_path: str) -> None:
        """
        Save the current state to a file.

        Args:
            file_path (str): Path to the file where the state will be saved.
        """
        data = {
            'entities': self.entities,
            'automatons': self.automatons,
            '_needs_building': self._needs_building,
            'ignore_list': self.ignore_list
        }
        joblib.dump(data, file_path)

    def load(self, file_path: str) -> None:
        """
        Load the state from a file.

        Args:
            file_path (str): Path to the file from which the state will be loaded.
        """
        data = joblib.load(file_path)
        self.entities = data['entities']
        self.automatons = data['automatons']
        self._needs_building = data['_needs_building']
        self.ignore_list = data['ignore_list']

    def one_hot_encode(self, text):
        labels = self.labels
        vec = [0 for _ in labels]
        for k, v in self.match(text):
            idx = labels.index(k)
            vec[idx] = 1
        return vec


if __name__ == "__main__":
    kw = KeywordFeatures()

    # Register some example entities
    kw.register_entity('fruit', ['apple', 'banana', 'cherry'])
    kw.register_entity('color', ['red', 'green', 'blue'])

    print(kw.one_hot_encode("red apple"))
    print(kw.one_hot_encode("not banana"))
    print(kw.one_hot_encode("green not"))

    # Check output
    print("Labels:", kw.labels)
    print("Match 'I have a red apple':", list(kw.match('I have a red apple')))
    print("Extract 'I have a red apple':", kw.extract('I have a red apple'))

    # Save to file
    kw.save('keyword_features.pkl')

    # Load from file
    kw_loaded = KeywordFeatures()
    kw_loaded.load('keyword_features.pkl')

    # Check output after loading
    print("Labels after loading:", kw_loaded.labels)
    print("Match 'I have a green banana':", list(kw_loaded.match('I have a green banana')))
    print("Extract 'I have a green banana':", kw_loaded.extract('I have a green banana'))
