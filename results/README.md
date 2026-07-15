# Benchmark results

Generated JSON reports belong in this directory and are ignored by Git by default.

Run all backends supported by your hardware, retain the raw reports as workflow artifacts, and copy the comparison table into the main README only after verifying that every backend used the same model revision, corpus, tokenizer, prompt set, seed, and generation settings.

Never compare perplexity measured on different generated texts. QuantLab evaluates a shared token sequence to keep the comparison meaningful.
