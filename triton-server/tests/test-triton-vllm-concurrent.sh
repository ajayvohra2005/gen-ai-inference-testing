#!/bin/bash

mkdir -p output
STARTTIME=$(date +%s)
for i in {1..8}
do
curl -X POST localhost:8080/v2/models/llama3-8b-instruct/generate \
-d '{"text_input": "<|begin_of_text|> <|start_header_id|>system<|end_header_id|> You are a technical writer. <|eot_id|> <|start_header_id|>user<|end_header_id|> Write a short article of less than 1000 words on Generative AI applications in Mobile apps. <|eot_id|> <|start_header_id|>assistant<|end_header_id|>", "sampling_parameters": "{ \"top_k\": 50, \"max_tokens\": 2048 }"  }' \
1>output/test-triton-vllm-concurrent.$i-0.out &
curl -X POST localhost:8080/v2/models/llama3-8b-instruct/generate \
-d '{"text_input": "<|begin_of_text|> <|start_header_id|>system<|end_header_id|> You are a technical writer. <|eot_id|> <|start_header_id|>user<|end_header_id|> Write a short article of less than 1000 words on Generative AI in computer visions. <|eot_id|> <|start_header_id|>assistant<|end_header_id|>", "sampling_parameters": "{ \"top_k\": 50, \"max_tokens\": 2048 }"  }' \
1>output/test-triton-vllm-concurrent.$i-1.out &
curl -X POST localhost:8080/v2/models/llama3-8b-instruct/generate \
-d '{"text_input": "<|begin_of_text|> <|start_header_id|>system<|end_header_id|> You are a technical writer. <|eot_id|> <|start_header_id|>user<|end_header_id|> Write a short article of less than 1000 words on the ethics of Generative AI. <|eot_id|> <|start_header_id|>assistant<|end_header_id|>", "sampling_parameters": "{ \"top_k\": 50, \"max_tokens\": 2048 }"  }' \
1>output/test-triton-vllm-concurrent.$i-2.out &
curl -X POST localhost:8080/v2/models/llama3-8b-instruct/generate \
-d '{"text_input": "<|begin_of_text|> <|start_header_id|>system<|end_header_id|> You are a technical writer. <|eot_id|> <|start_header_id|>user<|end_header_id|> Write a short article of less than 1000 words on the dangers of Generative AI. <|eot_id|> <|start_header_id|>assistant<|end_header_id|>", "sampling_parameters": "{ \"top_k\": 50, \"max_tokens\": 2048 }"  }' \
1>output/test-triton-vllm-concurrent.$i-3.out &
done

wait
ENDTIME=$(date +%s)
echo "Elapsed time: $((ENDTIME - STARTTIME)) seconds for 32 requests"