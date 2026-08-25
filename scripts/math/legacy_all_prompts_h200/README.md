# Legacy all-prompts Math evaluation

These launchers reproduce the pre-chunking Qwen3-8B Math evaluation protocol:

- five datasets: AIME24, AIME25, HMMT25, AMC23, and Minerva;
- 16 sampled solutions per problem;
- temperature 1.0, top-p 1.0, top-k -1;
- maximum 38,912 new tokens and model context 40,960;
- tensor parallel size 8;
- one `llm.generate(all_prompts, ...)` scheduler submission per dataset.

Fresh output roots are intentional. Existing sweep and grouped-OPSD directories
contain evaluations produced with eight-prompt chunking and must not be mixed
with this protocol.
