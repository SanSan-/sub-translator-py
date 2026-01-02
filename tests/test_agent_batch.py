import sub_translate.translators.agent as agent


def test_split_batch_output_returns_parts():
    separator = agent._build_batch_separator()
    texts = ["One", "Two\nTwo", "Three"]
    joined = agent._join_batch_texts(texts, separator)
    parts = agent._split_batch_output(joined, separator, len(texts))
    assert parts == texts


def test_split_batch_output_returns_none_on_mismatch():
    separator = agent._build_batch_separator()
    parts = agent._split_batch_output("One\nTwo", separator, 2)
    assert parts is None


def test_split_batch_output_allows_spaces_around_separator():
    separator = agent._build_batch_separator()
    output = f"First\n {separator} \nSecond"
    parts = agent._split_batch_output(output, separator, 2)
    assert parts == ["First", "Second"]