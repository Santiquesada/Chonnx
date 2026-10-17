
# chugin name
CHUGIN_NAME=Chonnx

# all of the c/cpp files that compose this chugin
C_MODULES=
CXX_MODULES=Chonnx.cpp

# where to find chugin.h
CK_SRC_PATH?=chuck/include

# where to install chugin
CHUGIN_PATH?=/usr/local/lib/chuck


# ---------------------------------------------------------------------------- #
# you won't generally need to change anything below this line for a new chugin #
# ---------------------------------------------------------------------------- #

# default target: print usage message and quit
current:
	@echo "[chugin build]: please use one of the following configurations:"
	@echo "   make linux, make mac, make web, or make win32"

ifneq ($(CK_TARGET),)
.DEFAULT_GOAL:=$(CK_TARGET)
ifeq ($(MAKECMDGOALS),)
MAKECMDGOALS:=$(.DEFAULT_GOAL)
endif
endif

.PHONY: mac osx linux linux-pulse linux-oss linux-jack linux-alsa win32
mac osx linux linux-pulse linux-oss linux-jack linux-alsa: all

win32:
	make -f makefile.win

CC=gcc
CXX=gcc
LD=g++

ifneq (,$(strip $(filter mac osx,$(MAKECMDGOALS))))
include makefile.mac
endif

ifneq (,$(strip $(filter linux,$(MAKECMDGOALS))))
include makefile.linux
endif

ifneq (,$(strip $(filter linux-oss,$(MAKECMDGOALS))))
include makefile.linux
endif

ifneq (,$(strip $(filter linux-pulse,$(MAKECMDGOALS))))
include makefile.linux
endif

ifneq (,$(strip $(filter linux-jack,$(MAKECMDGOALS))))
include makefile.linux
endif

ifneq (,$(strip $(filter linux-alsa,$(MAKECMDGOALS))))
include makefile.linux
endif

FLAGS+= -std=c++17

ifneq ($(CHUCK_DEBUG),)
FLAGS+= -g
else
FLAGS+= -O3
endif

ifneq ($(CHUCK_STRICT),)
FLAGS+= -Werror
endif

# default: build a dynamic chugin
CK_CHUGIN_STATIC?=0

ifeq ($(CK_CHUGIN_STATIC),0)
SUFFIX=.chug
FLAGS+= -I deps/onnxruntime/include/
else
SUFFIX=.schug
FLAGS+= -D__CK_DLL_STATIC__
endif
# webchugin extension
WEBSUFFIX=.wasm

C_OBJECTS=$(addsuffix .o,$(basename $(C_MODULES)))
CXX_OBJECTS=$(addsuffix .o,$(basename $(CXX_MODULES)))

CHUG=$(addsuffix $(SUFFIX),$(CHUGIN_NAME))
WEBCHUG=$(addsuffix $(WEBSUFFIX),$(CHUG))

all: deps/onnxruntime $(CHUG)

$(CHUG): $(C_OBJECTS) $(CXX_OBJECTS)
ifeq ($(CK_CHUGIN_STATIC),0)
	$(LD) $(LDFLAGS) -o $@ $^
else
	ar rv $@ $^
	ranlib $@
endif

$(C_OBJECTS): %.o: %.c
	$(CC) $(FLAGS) -c -o $@ $<

$(CXX_OBJECTS): %.o: %.cpp $(CK_SRC_PATH)/chugin.h
	$(CXX) $(FLAGS) -c -o $@ $<

# download and load onnxruntime
deps/onnxruntime:
	mkdir -p deps
	curl -o deps/onnxruntime.tgz -L https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-osx-arm64-1.31.0.tgz
	cd deps && tar -xvzf onnxruntime.tgz
	mv deps/onnxruntime-* $@
	touch $@
	rm deps/onnxruntime.tgz

# build as webchugin
web:
	emcc -O3 -s SIDE_MODULE=1 -s DISABLE_EXCEPTION_CATCHING=0 -fPIC -Wformat=0 	-I $(CK_SRC_PATH) $(CXX_MODULES) $(C_MODULES) -o $(WEBCHUG)

install: $(CHUG)
	mkdir -p $(CHUGIN_PATH)
	cp $^ $(CHUGIN_PATH)
	chmod 755 $(CHUGIN_PATH)/$(CHUG)

clean:
	rm -rf $(C_OBJECTS) $(CXX_OBJECTS) $(CHUG) $(WEBCHUG) deps Release Debug
