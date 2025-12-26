#!/bin/bash

# Change to parent directory
cd ..

for layer in {1..33}
do
	python save.py --layer $layer --data_subfolder refusal --output_dir "directions"
done
