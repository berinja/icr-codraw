#!/bin/sh

python3 main.py
python3 main.py -n_epochs 50
python3 main.py -filter
python3 main.py -only_until_peek
python3 main.py -filter -only_until_peek
python3 main.py -no_context
python3 main.py -no_image
python3 main.py -no_msg
python3 main.py -no_context -no_image
python3 main.py -no_msg -no_image
python3 main.py -no_context -no_msg
python3 main.py -no_context -no_image -no_msg
python3 main.py -text_pretrained all-mpnet-base-v2-noteller
python3 main.py -text_pretrained all-mpnet-base-v2-nodrawer

python3 main.py -task teller
python3 main.py -filter -task teller
python3 main.py -only_until_peek -task teller
python3 main.py -filter -only_until_peek -task teller
python3 main.py -no_context -task teller
python3 main.py -no_image -task teller
python3 main.py -no_msg -task teller
python3 main.py -no_context -no_image -task teller
python3 main.py -no_msg -no_image -task teller
python3 main.py -no_context -no_msg -task teller
python3 main.py -no_context -no_image -no_msg -task teller
python3 main.py -text_pretrained all-mpnet-base-v2-noteller -task teller
python3 main.py -text_pretrained all-mpnet-base-v2-nodrawer -task teller



