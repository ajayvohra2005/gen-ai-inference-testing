#!/bin/bash

mkdir -p output
STARTTIME=$(date +%s)
curl -X POST localhost:8080/v2/models/llama3-8b-instruct/generate \
-d '{"text_input": "<|begin_of_text|> <|start_header_id|>system<|end_header_id|> You are a technical writer. <|eot_id|> <|start_header_id|>user<|end_header_id|> Write a short article of less than 1000 words on Generative AI. <|eot_id|> <|start_header_id|>assistant<|end_header_id|>", "sampling_parameters": "{ \"top_k\": 50, \"sequence_length\": 2048 }"  }' \
1>output/test-triton.out
ENDTIME=$(date +%s)
echo "Elapsed time: $((ENDTIME - STARTTIME)) seconds"
