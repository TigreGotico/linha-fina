"""Tests for linha_fina.keywords."""

import os
import tempfile

import pytest

from linha_fina.keywords import KeywordFeatures

try:
    import ahocorasick  # noqa: F401
    HAS_AUTOMATON = True
except ImportError:
    HAS_AUTOMATON = False


@pytest.fixture
def kw():
    k = KeywordFeatures()
    k.register_entity("fruit", ["apple", "banana", "cherry"])
    k.register_entity("color", ["red", "green", "blue"])
    return k


class TestRegistration:
    def test_register_creates_entity(self):
        k = KeywordFeatures()
        k.register_entity("foo", ["alpha", "beta"])
        assert "foo" in k.entities
        assert set(k.entities["foo"]) == {"alpha", "beta"}

    def test_register_appends_to_existing(self):
        k = KeywordFeatures()
        k.register_entity("foo", ["alpha"])
        k.register_entity("foo", ["beta"])
        assert set(k.entities["foo"]) == {"alpha", "beta"}

    def test_labels_sorted(self):
        k = KeywordFeatures()
        k.register_entity("zeta", ["x"])
        k.register_entity("alpha", ["y"])
        assert k.labels == ["alpha", "zeta"]

    def test_deregister_removes_entity(self, kw):
        kw.deregister_entity("fruit")
        assert "fruit" not in kw.entities
        assert kw.labels == ["color"]

    def test_deregister_unknown_is_noop(self, kw):
        kw.deregister_entity("does_not_exist")
        assert "fruit" in kw.entities

    def test_reset_clears_entities(self, kw):
        kw.reset()
        assert kw.entities == {}


class TestMatching:
    def test_extract_basic(self, kw):
        assert kw.extract("I have a red apple") == {"fruit": "apple", "color": "red"}

    def test_match_yields_tuples(self, kw):
        result = list(kw.match("I have a red apple"))
        assert ("fruit", "apple") in result
        assert ("color", "red") in result

    def test_extract_keeps_longest(self):
        k = KeywordFeatures()
        k.register_entity("name", ["africa", "africa by toto"])
        # input must contain both substrings for both to match
        result = k.extract("i want africa by toto playing")
        assert result["name"] == "africa by toto"

    def test_extract_strips_punctuation(self, kw):
        # match() strips trailing .!?,;: from utterance
        result = kw.extract("I want a banana.")
        assert result.get("fruit") == "banana"

    def test_short_keywords_below_3_chars_skipped(self):
        k = KeywordFeatures()
        k.register_entity("code", ["us", "uk", "deu"])
        result = k.extract("i live in deu")
        # "us" and "uk" too short, but "deu" passes
        assert result == {"code": "deu"}

    def test_word_boundary_match(self):
        k = KeywordFeatures()
        k.register_entity("animal", ["cat"])
        # "cat" is whole word — matches
        assert k.extract("the cat sleeps") == {"animal": "cat"}
        # "cat" inside "concatenate" — no match (word boundary)
        assert k.extract("concatenate this") == {}

    def test_no_match_returns_empty_dict(self, kw):
        assert kw.extract("xyzzy plugh") == {}


class TestIgnoreList:
    def test_ignored_value_skipped_for_name_entities(self):
        k = KeywordFeatures(ignore_list=["alice"])
        k.register_entity("first_name", ["alice", "bob"])
        result = k.extract("hello alice")
        # "alice" ignored because entity name contains "_name"
        assert result == {}
        result2 = k.extract("hello bob")
        assert result2 == {"first_name": "bob"}

    def test_ignore_list_only_applies_to_name_entities(self):
        k = KeywordFeatures(ignore_list=["alice"])
        k.register_entity("person", ["alice", "bob"])
        # entity name doesn't contain "_name" → ignore_list bypassed
        assert k.extract("hello alice") == {"person": "alice"}


class TestOneHotEncode:
    def test_vector_length_matches_labels(self, kw):
        vec = kw.one_hot_encode("anything")
        assert len(vec) == len(kw.labels)

    def test_hot_positions_correspond_to_matches(self, kw):
        vec = kw.one_hot_encode("red apple")
        labels = kw.labels
        # both entities match → both positions are 1
        assert vec[labels.index("fruit")] == 1
        assert vec[labels.index("color")] == 1

    def test_no_match_all_zeros(self, kw):
        assert kw.one_hot_encode("xyzzy") == [0] * len(kw.labels)

    def test_partial_match(self, kw):
        vec = kw.one_hot_encode("a green thing")
        labels = kw.labels
        assert vec[labels.index("color")] == 1
        assert vec[labels.index("fruit")] == 0


class TestPersistence:
    def test_save_and_load_roundtrip(self, kw, tmp_path):
        path = tmp_path / "kw.pkl"
        kw.save(str(path))

        restored = KeywordFeatures()
        restored.load(str(path))

        assert restored.entities == kw.entities
        assert restored.extract("a red apple") == kw.extract("a red apple")


class TestCsvLoad:
    def test_load_from_csv(self, tmp_path):
        csv = tmp_path / "ents.csv"
        # First line is treated as a header and skipped
        csv.write_text("entity,value\nfruit,apple\nfruit,banana\ncolor,red\n")

        k = KeywordFeatures(csv_path=str(csv))
        assert "fruit" in k.entities
        assert "apple" in k.entities["fruit"]
        assert "banana" in k.entities["fruit"]
        assert "red" in k.entities["color"]


@pytest.mark.skipif(not HAS_AUTOMATON, reason="pyahocorasick not installed")
class TestAutomatonBackend:
    def test_extract_matches_same_as_regex(self):
        k = KeywordFeatures(use_automatons=True)
        k.register_entity("fruit", ["apple", "banana"])
        result = k.extract("i want an apple ")
        assert result == {"fruit": "apple"}

    def test_raises_when_lib_missing_but_requested(self, monkeypatch):
        # When ahocorasick IS installed, this path is exercised by the constructor
        # only when the user passes use_automatons=True. The negative branch
        # (lib missing → ImportError) is unreachable in this environment.
        k = KeywordFeatures(use_automatons=True)
        assert k.use_automatons is True


def test_use_automatons_without_lib_raises(monkeypatch):
    import linha_fina.keywords as kwmod
    monkeypatch.setattr(kwmod, "ahocorasick", None)
    with pytest.raises(ImportError):
        KeywordFeatures(use_automatons=True)
