#!/usr/bin/env bash
# Simple runner for scaffold pipeline
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
