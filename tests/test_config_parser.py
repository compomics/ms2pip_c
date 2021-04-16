import os

from ms2pip.config_parser import ConfigParser


TEST_DIR = os.path.dirname(__file__)


class TestConfigParser:
    def test_load_ms2pip_txt(self):
        config_file = os.path.join(TEST_DIR, "test_data/config.txt")
        config_parser = ConfigParser(config_file)
        config_parser.load()

        assert config_parser.config == {
            "ms2pip": {
                "ptm": [
                    "Oxidation,15.994915,opt,M",
                    "Carbamidomethyl,57.021464,opt,C",
                    "Acetyl,42.010565,opt,N-term",
                ],
                "sptm": [],
                "gptm": [],
                "frag_method": "HCD",
                "frag_error": 0.02,
                "out": "csv",
            }
        }

    def test_load_toml(self):
        config_file = os.path.join(TEST_DIR, "test_data/config.toml")
        config_parser = ConfigParser(config_file)
        config_parser.load()

        assert config_parser.config == {
            "ms2pip": {
                "ptm": [
                    "Oxidation,15.994915,opt,M",
                    "Carbamidomethyl,57.021464,opt,C",
                    "Acetyl,42.010565,opt,N-term",
                ],
                "sptm": [],
                "gptm": [],
                "frag_method": "HCD",
                "frag_error": 0.02,
                "out": "csv",
            }
        }

    def test_write_toml(self):
        test_config_file = os.path.join(TEST_DIR, "test_data/config_test.toml")
        config_parser = ConfigParser()
        target_config = {
            "ms2pip": {
                "ptm": [
                    "Oxidation,15.994915,opt,M",
                    "Carbamidomethyl,57.021464,opt,C",
                    "Acetyl,42.010565,opt,N-term",
                ],
                "sptm": [],
                "gptm": [],
                "frag_method": "HCD",
                "frag_error": 0.02,
                "out": "csv",
            }
        }
        config_parser.config = target_config
        config_parser.write(test_config_file)
        config_parser.load()
        os.remove(test_config_file)
        assert config_parser.config == target_config
