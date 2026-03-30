# tests/test_extract_params.py
import pytest
from research.extract_params import extract_params


class TestExtractParams:
    def test_extracts_millions_params(self):
        text = "We train a 50M parameter transformer"
        result = extract_params(text)
        assert result["params"] == 50_000_000

    def test_extracts_billions_params(self):
        text = "Our 1.3B param model achieves"
        result = extract_params(text)
        assert result["params"] == 1_300_000_000

    def test_extracts_int_bitwidth(self):
        text = "Using int4 quantization we compress"
        result = extract_params(text)
        assert result["bits"] == 4

    def test_extracts_dash_bitwidth(self):
        text = "We apply 6-bit symmetric quantization"
        result = extract_params(text)
        assert result["bits"] == 6

    def test_extracts_both(self):
        text = "A 23M param model with int6 QAT fits in 12MB"
        result = extract_params(text)
        assert result["params"] == 23_000_000
        assert result["bits"] == 6

    def test_returns_none_for_missing_params(self):
        text = "We propose a novel attention mechanism"
        result = extract_params(text)
        assert result["params"] is None

    def test_returns_none_for_missing_bits(self):
        text = "A 20M parameter model"
        result = extract_params(text)
        assert result["bits"] is None

    def test_extracts_decimal_params(self):
        text = "The 1.5M model uses"
        result = extract_params(text)
        assert result["params"] == 1_500_000

    def test_extracts_w4a8_format(self):
        text = "Using W4A8 mixed precision"
        result = extract_params(text)
        assert result["bits"] == 4

    def test_returns_empty_for_empty_string(self):
        result = extract_params("")
        assert result["params"] is None
        assert result["bits"] is None

    def test_million_word_form(self):
        text = "We train 20 million parameter models"
        result = extract_params(text)
        assert result["params"] == 20_000_000
