CC=gcc
CFLAGS=-c -Wall -O3
LDFLAGS=

MAKE_DIR = $(PWD)
SOURCE_DIR = $(MAKE_DIR)/src
TEST_DIR = $(MAKE_DIR)/test

INC_SRCH_PATH :=
INC_SRCH_PATH += $(SOURCE_DIR)