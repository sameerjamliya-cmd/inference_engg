from bpe_tokenizer import BPETokenizer


def check_roundtrip(tokenizer, text):
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    status = "OK" if decoded == text else "FAIL"
    print(f"[{status}] {text!r} -> {len(ids)} tokens -> {decoded!r}")
    assert decoded == text


if __name__ == "__main__":
    tokenizer = BPETokenizer()

    corpus = (
        "the quick brown fox jumps over the lazy dog. "
        "the dog barks at the fox. the fox runs away. "
    ) * 20

    final_train_ids = tokenizer.train(corpus, num_merges=50)
    print(f"learned {len(tokenizer.merges)} merges")
    print(f"training corpus: {len(corpus.encode('utf-8'))} bytes -> {len(final_train_ids)} tokens after merges")
    print()

    # roundtrip on training-distribution text
    check_roundtrip(tokenizer, "the quick fox")

    # roundtrip on text NOT seen during training
    check_roundtrip(tokenizer, "hello world, this sentence was never in the corpus")

    # roundtrip on unicode / emoji, also never seen during training
    check_roundtrip(tokenizer, "unicode test: héllo wörld, 日本語, 👋🚀🔥")

    # edge case: empty string
    check_roundtrip(tokenizer, "")

    print()
    print("all roundtrip tests passed")
