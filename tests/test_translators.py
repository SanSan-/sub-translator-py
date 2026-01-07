import unittest
from unittest.mock import MagicMock, patch

from sub_translate.models import TranslationOptions
from sub_translate.translators.agent import AgentTranslator
from sub_translate.translators.fsm import FsmTranslator
from sub_translate.translators.google_web import GoogleWebTranslator
from sub_translate.translators.madlad import MadladTranslator
from sub_translate.translators.nllb import NllbTranslator
from sub_translate.translators.seamless import SeamlessTranslator


def _setup_hf_mocks(mock_load):
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_device = MagicMock()
    mock_load.return_value = (mock_tokenizer, mock_model, mock_device)

    mock_tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 5))}
    mock_tokenizer.decode.return_value = "Перевод"
    mock_model.generate.return_value = [MagicMock()]
    return mock_tokenizer, mock_model, mock_device


class TestTranslators(unittest.TestCase):
    def setUp(self):
        self.texts = ["Hello", "World"]
        self.options = TranslationOptions(source_lang="en", target_lang="ru")

    def test_google_web_translator(self):
        with patch("sub_translate.translators.google_web._translate") as mock_translate:
            # Mock _translate to return a dict mapping index to translated text
            mock_translate.side_effect = lambda text, opts, timeout: {
                k: f"Translated {v}" for k, v in text.items()
            }

            translator = GoogleWebTranslator()
            result = translator.translate_batch(self.texts, self.options)

            self.assertEqual(result, ["Translated Hello", "Translated World"])
            mock_translate.assert_called()

    def test_agent_translator(self):
        with patch("sub_translate.translators.agent._request_translation") as mock_request, \
             patch("sub_translate.translators.agent._build_batch_separator", return_value="<<<SEP>>>"), \
             patch("sub_translate.translators.agent._load_request_cache"), \
             patch("sub_translate.translators.agent._save_request_cache"), \
             patch("sub_translate.translators.agent._request_cache", {}):

            # Mock the response from the agent model
            mock_request.return_value = "Привет\n<<<SEP>>>\nМир"

            translator = AgentTranslator()
            result = translator.translate_batch(self.texts, self.options)

            self.assertEqual(result, ["Привет", "Мир"])
            mock_request.assert_called()

    @patch("sub_translate.translators.fsm.load_model_components")
    def test_fsm_translator(self, mock_load):
        _setup_hf_mocks(mock_load)

        translator = FsmTranslator()
        # FSM only supports en->ru, ensure options are correct
        options = TranslationOptions(source_lang="en", target_lang="ru")
        result = translator.translate_batch(self.texts, options)

        self.assertEqual(result, ["Перевод", "Перевод"])
        mock_load.assert_called()

    @patch("sub_translate.translators.madlad.load_model_components")
    def test_madlad_translator(self, mock_load):
        _setup_hf_mocks(mock_load)

        translator = MadladTranslator()
        result = translator.translate_batch(self.texts, self.options)

        self.assertEqual(result, ["Перевод", "Перевод"])
        mock_load.assert_called()

    @patch("sub_translate.translators.seamless.load_model_components")
    def test_seamless_translator(self, mock_load):
        _setup_hf_mocks(mock_load)

        translator = SeamlessTranslator()
        result = translator.translate_batch(self.texts, self.options)

        self.assertEqual(result, ["Перевод", "Перевод"])
        mock_load.assert_called()

    @patch("sub_translate.translators.nllb.load_model_components")
    def test_nllb_translator(self, mock_load):
        mock_tokenizer, mock_model, mock_device = _setup_hf_mocks(mock_load)

        # NLLB needs lang_code_to_id for forced_bos_token_id resolution
        mock_tokenizer.lang_code_to_id = {"rus_Cyrl": 123}

        translator = NllbTranslator()
        result = translator.translate_batch(self.texts, self.options)

        self.assertEqual(result, ["Перевод", "Перевод"])
        mock_load.assert_called()
