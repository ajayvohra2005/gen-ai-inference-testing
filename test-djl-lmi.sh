#!/bin/bash

mkdir -p output
STARTTIME=$(date +%s)
curl -X POST http://localhost:8080/invocations \
  --connect-timeout 120 \
  -H "Content-type: application/json" \
  -d '{"inputs":"<|begin_of_text|> <|start_header_id|>system<|end_header_id|> You are a technical writer. <|eot_id|> <|start_header_id|>user<|end_header_id|> Write a short article of less than 1000 words on Generative AI. <|eot_id|> <|start_header_id|>assistant<|end_header_id|>","parameters":{ "do_sample": true, "max_new_tokens": 2048, "top_k": 50 }}' \
1>output/test-djl-lmi.out
ENDTIME=$(date +%s)
echo "Elapsed time: $((ENDTIME - STARTTIME)) seconds"