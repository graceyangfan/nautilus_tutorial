#!/bin/bash

#set variables 
startDate="2022-01-01"
endDate="2023-07-09"
interval="5m"
store_dir="../example_data/"
# Read symbols from filtered_symbols.txt and gather them into an array
symbols=()
while IFS= read -r symbol
do
  # Remove the slash from the symbol
  symbol=$(echo "$symbol" | tr -d '/')
  symbols+=("$symbol")
done < filtered_symbols.txt


# Convert the symbols array to a string, shuffle it, and convert it back to an array
#IFS=$'\n' symbols=($(printf '%s\n' "${symbols[@]}" | shuf))
# Iterate over the symbols array
for symbol in "${symbols[@]}"
do
    #STORE_DIRECTORY=$store_dir python3 download_kline.py -s $symbol -t um -i $interval  -skip-monthly  1 -startDate $startDate -endDate $endDate
    #STORE_DIRECTORY=$store_dir python3 download_metrics.py -s $symbol -t um  -startDate $startDate -endDate $endDate
    python3 concat_df.py --symbol $symbol --interval $interval --store_dir $store_dir --startDate $startDate --endDate $endDate
done