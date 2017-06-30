#!/bin/bash

.PHONY: all


all: dev

dev:
	python test_correctness.py

test: correct speed capacity

correct:
	python test/test_correctness.py

speed: 
	python test/test_speed.py

capacity:
	python test/test_capacity.py
