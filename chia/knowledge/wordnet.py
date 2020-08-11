from nltk import downloader
from nltk.corpus import wordnet

from chia.knowledge.relation import RelationSource


class WordNetAccess(RelationSource):
    def __init__(self):
        # Check if required corpora are there
        corpus_downloader = downloader.Downloader()
        for corpus in ("wordnet", "omw"):
            if not corpus_downloader.is_installed(corpus):
                corpus_downloader.download(corpus)

        try:
            wordnet.synset("dog.n.01")
        except LookupError:
            raise ValueError("Could not perform test lookup!")

    def _get_hypernyms(self, synset):
        return {
            f"WordNet3.0::{hsynset.name()}"
            for hsynset in wordnet.synset(synset).hypernyms()
        }

    def _get_hyponyms(self, synset):
        return {
            f"WordNet3.0::{hsynset.name()}"
            for hsynset in wordnet.synset(synset).hyponyms()
        }

    def get_right_for(self, uid_left):
        if uid_left.startswith("WordNet3.0::"):
            return self._get_hypernyms(uid_left[12:])
        else:
            return set()

    def get_left_for(self, uid_right):
        if uid_right.startswith("WordNet3.0::"):
            return self._get_hyponyms(uid_right[12:])
        else:
            return set()
