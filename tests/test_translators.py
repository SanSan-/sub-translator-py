import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sub_translate.models import TranslationOptions
from sub_translate.translators.agent import AgentTranslator
from sub_translate.translators.base import TranslationError
from sub_translate.translators.google_web import GoogleWebTranslator
from sub_translate.translators.local import nllb as nllb_module
from sub_translate.translators.local.nllb import (
    Nllb600MTranslator,
    _resolve_forced_bos,
)
from sub_translate.utils.huggingface import ModelLoadOptions, ResolvedLocalModel

MODEL_SOURCE = ResolvedLocalModel(
    path=Path("mock-local-model"),
    revision="0123456789abcdef0123456789abcdef01234567",
)


def _setup_hf_mocks(mock_load):
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_device = MagicMock()
    mock_load.return_value = (mock_tokenizer, mock_model, mock_device)

    mock_tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 5))}
    mock_tokenizer.decode.return_value = "Перевод"
    mock_model.generate.return_value = [MagicMock()]
    return mock_tokenizer, mock_model, mock_device


def _assert_strict_local_load(mock_load) -> None:
    assert mock_load.call_args.args[0] == str(MODEL_SOURCE.path)
    assert mock_load.call_args.args[1] == MODEL_SOURCE.path
    options = mock_load.call_args.args[4]
    assert isinstance(options, ModelLoadOptions)
    assert options.revision == MODEL_SOURCE.revision
    assert options.local_files_only is True


class TestTranslators(unittest.TestCase):
    def setUp(self):
        self.texts = ["Hello", "World"]
        self.options = TranslationOptions(source_lang="en", target_lang="ru")

    def test_google_web_translator(self):
        with patch("sub_translate.translators.google_web._translate") as mock_translate:
            # Mock _translate to return a dict mapping index to translated text
            mock_translate.side_effect = lambda text, opts, timeout: {k: f"Translated {v}" for k, v in text.items()}

            translator = GoogleWebTranslator()
            result = translator.translate_batch(self.texts, self.options)

            self.assertEqual(result, ["Translated Hello", "Translated World"])
            mock_translate.assert_called()

    def test_google_web_translator_rejects_missing_batch_item(self):
        with patch(
            "sub_translate.translators.google_web._translate",
            return_value={"0": "Перевод"},
        ):
            translator = GoogleWebTranslator()

            with self.assertRaisesRegex(TranslationError, "не содержит перевод для каждого элемента"):
                translator.translate_batch(self.texts, self.options)

    def test_agent_translator(self):
        with (
            patch("sub_translate.translators.agent._request_translation") as mock_request,
            patch("sub_translate.translators.agent._build_batch_separator", return_value="<<<SEP>>>"),
            patch("sub_translate.translators.agent._load_request_cache"),
            patch("sub_translate.translators.agent._save_request_cache"),
            patch("sub_translate.translators.agent._request_cache", {}),
        ):
            # Mock the response from the agent model
            mock_request.return_value = "Привет\n<<<SEP>>>\nМир"

            translator = AgentTranslator()
            result = translator.translate_batch(self.texts, self.options)

            self.assertEqual(result, ["Привет", "Мир"])
            mock_request.assert_called()

    @patch("sub_translate.translators.local.nllb.load_model_components")
    def test_nllb_600m_translator_uses_pytorch_weights(self, mock_load):
        mock_tokenizer, _, _ = _setup_hf_mocks(mock_load)
        mock_tokenizer.lang_code_to_id = {"rus_Cyrl": 123}
        translator = Nllb600MTranslator()
        translator.unload()

        with patch(
            "sub_translate.translators.local.nllb.resolve_registered_model",
            return_value=MODEL_SOURCE,
        ) as resolve_model:
            result = translator.translate_batch(self.texts, self.options)

        self.assertEqual(result, ["Перевод", "Перевод"])
        resolve_model.assert_called_with("nllb-600m", self.options)
        load_options = mock_load.call_args.args[4]
        self.assertFalse(load_options.use_safetensors)
        self.assertFalse(load_options.enable_cpu_offload)
        _assert_strict_local_load(mock_load)
        translator.unload()

    @patch("sub_translate.translators.local.nllb.load_model_components")
    def test_nllb_enables_cpu_offload_only_after_explicit_consent(self, mock_load):
        mock_tokenizer, _, _ = _setup_hf_mocks(mock_load)
        mock_tokenizer.lang_code_to_id = {"rus_Cyrl": 123}
        options = TranslationOptions(
            source_lang="en",
            target_lang="ru",
            allow_cpu_fallback=True,
        )
        translator = Nllb600MTranslator()
        translator.unload()

        with patch(
            "sub_translate.translators.local.nllb.resolve_registered_model",
            return_value=MODEL_SOURCE,
        ):
            translator.translate_batch(self.texts, options)

        self.assertTrue(mock_load.call_args.args[4].enable_cpu_offload)
        translator.unload()

    def test_nllb_oom_unloads_model_without_cpu_fallback(self):
        translator = Nllb600MTranslator()
        with (
            patch.object(nllb_module._ENGINE, "ensure_loaded"),
            patch.object(
                nllb_module._ENGINE,
                "translate_text",
                side_effect=nllb_module.torch.cuda.OutOfMemoryError("CUDA out of memory"),
            ),
            patch.object(nllb_module._ENGINE, "unload") as unload,
            self.assertRaisesRegex(TranslationError, "скрытый переход на CPU не выполнялся"),
        ):
            translator.translate_batch(["Hello"], self.options)

        unload.assert_called_once_with()


def test_nllb_resolves_target_token_with_transformers_v5_tokenizer() -> None:
    tokenizer = MagicMock(spec=["convert_tokens_to_ids", "unk_token_id"])
    tokenizer.convert_tokens_to_ids.return_value = 321
    tokenizer.unk_token_id = 0

    assert _resolve_forced_bos(tokenizer, "rus_Cyrl") == 321
    tokenizer.convert_tokens_to_ids.assert_called_once_with("rus_Cyrl")
