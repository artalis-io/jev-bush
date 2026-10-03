CC ?= cc
AR ?= ar
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig

CFLAGS ?= -O3
WARNFLAGS ?= -std=c11 -Wall -Wextra -pedantic
OMPFLAGS ?=
LDLIBS ?= -lm
SOFLAGS = -fPIC -fvisibility=hidden -DJB_NO_MAIN -DJB_SHARED -DJB_BUILD_SHARED

UNAME_S := $(shell uname -s 2>/dev/null)
HARDEN_CFLAGS = -fstack-protector-strong
ifeq ($(UNAME_S),Darwin)
SHARED_NAME = libjb.dylib
SHARED_REAL = libjb.0.2.0.dylib
SHARED_FLAGS = -dynamiclib -Wl,-install_name,$(LIBDIR)/libjb.dylib
else
SHARED_NAME = libjb.so
SHARED_REAL = libjb.so.0.2.0
SHARED_FLAGS = -shared -Wl,-soname,libjb.so.0
HARDEN_CFLAGS += -D_FORTIFY_SOURCE=3
HARDEN_LDFLAGS = -Wl,-z,relro,-z,now,-z,noexecstack
endif

JB_CFLAGS = $(CFLAGS) $(WARNFLAGS) $(HARDEN_CFLAGS)
LDFLAGS += $(HARDEN_LDFLAGS)

.PHONY: all clean check debug sanitize thread-sanitize install uninstall

all: jb libjb.a $(SHARED_NAME)

jb: jb.c jb.h
	$(CC) $(CPPFLAGS) $(JB_CFLAGS) $(OMPFLAGS) jb.c $(LDFLAGS) $(OMPFLAGS) $(LDLIBS) -o $@

jb.o: jb.c jb.h
	$(CC) $(CPPFLAGS) $(JB_CFLAGS) $(OMPFLAGS) -DJB_NO_MAIN -c jb.c -o $@

jb.pic.o: jb.c jb.h
	$(CC) $(CPPFLAGS) $(JB_CFLAGS) $(OMPFLAGS) $(SOFLAGS) -c jb.c -o $@

libjb.a: jb.o
	$(AR) rcs $@ $<

$(SHARED_REAL): jb.pic.o
	$(CC) $(SHARED_FLAGS) $(LDFLAGS) $(OMPFLAGS) $< $(LDLIBS) -o $@

$(SHARED_NAME): $(SHARED_REAL)
	ln -sf $(SHARED_REAL) $(SHARED_NAME)
	@if test "$(UNAME_S)" != Darwin; then ln -sf $(SHARED_REAL) libjb.so.0; fi

test-library-faults: test/library_faults.c libjb.a jb.h
	$(CC) $(CPPFLAGS) $(JB_CFLAGS) -I. test/library_faults.c libjb.a $(LDFLAGS) $(OMPFLAGS) $(LDLIBS) -o $@

test-library-api: test/library_api.c libjb.a jb.h
	$(CC) $(CPPFLAGS) $(JB_CFLAGS) -I. test/library_api.c libjb.a $(LDFLAGS) $(OMPFLAGS) $(LDLIBS) -o $@

check: jb test-library-faults test-library-api
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

install: libjb.a $(SHARED_NAME) jb.h jev-bush.pc
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(INCLUDEDIR) $(DESTDIR)$(PKGCONFIGDIR)
	install -m 0644 libjb.a $(DESTDIR)$(LIBDIR)/libjb.a
	install -m 0755 $(SHARED_REAL) $(DESTDIR)$(LIBDIR)/$(SHARED_REAL)
	ln -sf $(SHARED_REAL) $(DESTDIR)$(LIBDIR)/$(SHARED_NAME)
	@if test "$(UNAME_S)" != Darwin; then ln -sf $(SHARED_REAL) $(DESTDIR)$(LIBDIR)/libjb.so.0; fi
	install -m 0644 jb.h $(DESTDIR)$(INCLUDEDIR)/jb.h
	install -m 0644 jev-bush.pc $(DESTDIR)$(PKGCONFIGDIR)/jev-bush.pc

uninstall:
	rm -f $(DESTDIR)$(LIBDIR)/libjb.a $(DESTDIR)$(LIBDIR)/$(SHARED_NAME) \
	  $(DESTDIR)$(LIBDIR)/$(SHARED_REAL) $(DESTDIR)$(LIBDIR)/libjb.so.0
	rm -f $(DESTDIR)$(INCLUDEDIR)/jb.h $(DESTDIR)$(PKGCONFIGDIR)/jev-bush.pc

clean:
	rm -f jb jb.o jb.pic.o libjb.a libjb.so libjb.so.0 libjb.so.0.2.0 \
	  libjb.dylib libjb.0.2.0.dylib test-library-faults test-library-api
