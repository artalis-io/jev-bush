CC ?= cc
AR ?= ar
.DEFAULT_GOAL := all
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig

CFLAGS ?= -O3
WARNFLAGS ?= -std=c11 -Wall -Wextra -pedantic
OMPFLAGS ?=
LDLIBS ?= -lm
SOFLAGS = -fPIC -fvisibility=hidden -DJB_NO_MAIN -DJB_SHARED -DJB_BUILD_SHARED
PRIVATE_LIBS = -lm

ifneq ($(findstring -fopenmp,$(OMPFLAGS)),)
PRIVATE_LIBS += $(OMPFLAGS)
endif
ifneq ($(findstring -DJB_CUDA,$(CPPFLAGS) $(CFLAGS)),)
PRIVATE_LIBS += -ldl -lpthread
endif

TARGET_OS ?= $(shell uname -s 2>/dev/null)
HARDEN_CFLAGS = -fstack-protector-strong
ifeq ($(TARGET_OS),Darwin)
SHARED_NAME = libjb.dylib
SHARED_REAL = libjb.0.3.0.dylib
SHARED_FLAGS = -dynamiclib -Wl,-install_name,$(LIBDIR)/libjb.dylib
else
SHARED_NAME = libjb.so
SHARED_REAL = libjb.so.0.3.0
SHARED_FLAGS = -shared -Wl,-soname,libjb.so.0
HARDEN_CFLAGS += -D_FORTIFY_SOURCE=3
HARDEN_LDFLAGS = -Wl,-z,relro,-z,now,-z,noexecstack
endif

JB_CFLAGS = $(CFLAGS) $(WARNFLAGS) $(HARDEN_CFLAGS)
LDFLAGS += $(HARDEN_LDFLAGS)

JB_MODULES = src/jb.c src/foundation.inc src/json.inc src/model.inc \
	src/kernels_scalar.inc src/kernels_avx2.inc \
	src/kernels_avx512.inc src/kernels_neon.inc src/kernels_dispatch.inc src/cuda.inc \
	src/engine.inc src/decision.inc src/api.inc src/cli_tests.inc

.PHONY: all clean check check-amalgamation debug format install install-hooks lint sanitize \
	thread-sanitize uninstall FORCE

FORCE:

.build:
	mkdir -p $@

# Compiler-option changes are build inputs. Preserve the timestamp when the
# configuration is identical so ordinary incremental builds remain cheap.
.build/config: FORCE | .build
	@printf '%s\n' 'CC=$(CC)' 'CPPFLAGS=$(CPPFLAGS)' 'JB_CFLAGS=$(JB_CFLAGS)' \
	  'OMPFLAGS=$(OMPFLAGS)' 'LDFLAGS=$(LDFLAGS)' 'LDLIBS=$(LDLIBS)' \
	  'SOFLAGS=$(SOFLAGS)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

.build/jev-bush.pc: jev-bush.pc.in .build/config | .build
	sed -e 's|@VERSION@|0.3.0|g' -e 's|@PRIVATE_LIBS@|$(PRIVATE_LIBS)|g' $< > $@

all: jb libjb.a $(SHARED_NAME)

jb.c: $(JB_MODULES) tools/amalgamate.py
	python3 tools/amalgamate.py

check-amalgamation:
	python3 tools/amalgamate.py --check
	python3 tools/check_error_boundaries.py jb.c jb.h
	python3 tools/check_parallel_regions.py jb.c

format:
	tools/format.sh

lint:
	tools/lint.sh

install-hooks:
	git config core.hooksPath .githooks

jb: jb.c jb.h .build/config
	$(CC) $(CPPFLAGS) $(JB_CFLAGS) $(OMPFLAGS) jb.c $(LDFLAGS) $(OMPFLAGS) $(LDLIBS) -o $@

jb.o: jb.c jb.h .build/config
	$(CC) $(CPPFLAGS) $(JB_CFLAGS) $(OMPFLAGS) -DJB_NO_MAIN -c jb.c -o $@

jb.pic.o: jb.c jb.h .build/config
	$(CC) $(CPPFLAGS) $(JB_CFLAGS) $(OMPFLAGS) $(SOFLAGS) -c jb.c -o $@

libjb.a: jb.o
	$(AR) rcs $@ $<

$(SHARED_REAL): jb.pic.o
	$(CC) $(SHARED_FLAGS) $(LDFLAGS) $(OMPFLAGS) $< $(LDLIBS) -o $@

$(SHARED_NAME): $(SHARED_REAL)
	ln -sf $(SHARED_REAL) $(SHARED_NAME)
	@if test "$(TARGET_OS)" != Darwin; then ln -sf $(SHARED_REAL) libjb.so.0; fi

test-library-faults: test/library_faults.c libjb.a jb.h
	$(CC) $(CPPFLAGS) $(JB_CFLAGS) -I. test/library_faults.c libjb.a $(LDFLAGS) $(OMPFLAGS) $(LDLIBS) -o $@

test-library-api: test/library_api.c libjb.a jb.h
	$(CC) $(CPPFLAGS) $(JB_CFLAGS) -I. test/library_api.c libjb.a $(LDFLAGS) $(OMPFLAGS) $(LDLIBS) -o $@

check: check-amalgamation jb test-library-faults test-library-api
	./jb --selftest
	./test-library-faults
	./test-library-api
	sh tools/check_requests.sh ./jb

debug:
	$(MAKE) clean
	$(MAKE) CFLAGS='-O0 -g -fno-omit-frame-pointer' WARNFLAGS='-std=c11 -Wall -Wextra -pedantic -Wshadow -Wformat=2 -Werror' HARDEN_CFLAGS= check

sanitize:
	$(MAKE) clean
	$(MAKE) CC=clang CFLAGS='-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all' WARNFLAGS='-std=c11 -Wall -Wextra -pedantic -Wshadow -Wformat=2 -Werror' HARDEN_CFLAGS= LDFLAGS='-fsanitize=address,undefined' check

thread-sanitize:
	$(MAKE) clean
	$(MAKE) CC=clang CFLAGS='-O1 -g -fno-omit-frame-pointer -fsanitize=thread -fno-sanitize-recover=all' WARNFLAGS='-std=c11 -Wall -Wextra -pedantic -Wshadow -Wformat=2 -Werror' HARDEN_CFLAGS= LDFLAGS='-fsanitize=thread' test-library-faults test-library-api
	./test-library-faults
	./test-library-api

install: libjb.a $(SHARED_NAME) jb.h .build/jev-bush.pc
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(INCLUDEDIR) $(DESTDIR)$(PKGCONFIGDIR)
	install -m 0644 libjb.a $(DESTDIR)$(LIBDIR)/libjb.a
	install -m 0755 $(SHARED_REAL) $(DESTDIR)$(LIBDIR)/$(SHARED_REAL)
	ln -sf $(SHARED_REAL) $(DESTDIR)$(LIBDIR)/$(SHARED_NAME)
	@if test "$(TARGET_OS)" != Darwin; then ln -sf $(SHARED_REAL) $(DESTDIR)$(LIBDIR)/libjb.so.0; fi
	install -m 0644 jb.h $(DESTDIR)$(INCLUDEDIR)/jb.h
	install -m 0644 .build/jev-bush.pc $(DESTDIR)$(PKGCONFIGDIR)/jev-bush.pc

uninstall:
	rm -f $(DESTDIR)$(LIBDIR)/libjb.a $(DESTDIR)$(LIBDIR)/$(SHARED_NAME) \
	  $(DESTDIR)$(LIBDIR)/$(SHARED_REAL) $(DESTDIR)$(LIBDIR)/libjb.so.0
	rm -f $(DESTDIR)$(INCLUDEDIR)/jb.h $(DESTDIR)$(PKGCONFIGDIR)/jev-bush.pc

clean:
	rm -f jb jb.o jb.pic.o libjb.a libjb.so libjb.so.0 libjb.so.0.2.0 libjb.so.0.3.0 \
	  libjb.dylib libjb.0.2.0.dylib libjb.0.3.0.dylib test-library-faults test-library-api
	rm -rf .build
