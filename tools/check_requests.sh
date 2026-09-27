#!/bin/sh
# Feed malformed and adversarial requests to `jb --check-request`.
# Valid requests must exit 0; every bad request must be rejected by jb's own
# validation (exit 2 with a "jb:" message), never by a crash or a sanitizer.
# Usage: tools/check_requests.sh ./jb
set -eu

JB=${1:-./jb}
DIR=$(mktemp -d)
trap 'rm -rf "$DIR"' EXIT
fail=0

expect() {
    want=$1 name=$2
    set +e
    "$JB" --check-request "$DIR/$name" >"$DIR/out" 2>"$DIR/err"
    got=$?
    set -e
    if [ "$got" -ne "$want" ] || grep -q Sanitizer "$DIR/err" ||
        { [ "$want" -eq 2 ] && ! grep -q '^jb: ' "$DIR/err"; }; then
        echo "FAIL $name: exit $got, want $want"
        head -c 2000 "$DIR/err"
        fail=1
    else
        echo "ok   $name: $(grep -h -e '^jb: ' -e '^{' "$DIR/err" "$DIR/out" | head -n 1)"
    fi
}

Q='"questions":{"q":{"type":"noul"}}'

cp examples/request.json "$DIR/example.json"
expect 0 example.json
printf '{"st\\u0061te":"x",%s}' "$Q" >"$DIR/escaped-key.json"
expect 0 escaped-key.json

# Trailing comma at the very end of the buffer (no newline): object key read.
printf '{"state":"x",%s,' "$Q" >"$DIR/trailing-comma.json"
expect 2 trailing-comma.json

# Deep nesting must be rejected before it exhausts the stack.
{ printf '{"state":'; head -c 200000 /dev/zero | tr '\0' '['; } >"$DIR/deep.json"
expect 2 deep.json

# Score criteria past the candidate array capacity.
{
    printf '{"state":"x","questions":{"q":{"type":"score","criteria":[0'
    i=1
    while [ $i -lt 400 ]; do printf ',%d' $i; i=$((i + 1)); done
    printf ']}}}'
} >"$DIR/score-overflow.json"
expect 2 score-overflow.json

{
    printf '{"state":"x","questions":{"q":{"type":"choice","criteria":{"k0":0'
    i=1
    while [ $i -lt 400 ]; do printf ',"k%d":0' $i; i=$((i + 1)); done
    printf '}}}}'
} >"$DIR/choice-overflow.json"
expect 2 choice-overflow.json

printf '{"state":"\\\000",%s}' "$Q" >"$DIR/escape-nul.json"
expect 2 escape-nul.json
printf '{"state":"a\\u0000b",%s}' "$Q" >"$DIR/unicode-nul.json"
expect 2 unicode-nul.json
printf '{"state":"\\ud800",%s}' "$Q" >"$DIR/lone-surrogate.json"
expect 2 lone-surrogate.json
printf '{"state":"\377",%s}' "$Q" >"$DIR/bad-utf8.json"
expect 2 bad-utf8.json
printf '{"state":"x","state":"y",%s}' "$Q" >"$DIR/duplicate-key.json"
expect 2 duplicate-key.json
printf '{"state":"x","samples":0,%s}' "$Q" >"$DIR/samples-zero.json"
expect 2 samples-zero.json
printf '{"state":"x","samples":33,%s}' "$Q" >"$DIR/samples-large.json"
expect 2 samples-large.json
printf '{"state":"x","steps":2,%s}' "$Q" >"$DIR/steps.json"
expect 2 steps.json
printf '{"state":"x","questions":{"q":{"type":"other"}}}' >"$DIR/unknown-type.json"
expect 2 unknown-type.json
printf '{"state":"x","questions":"[1]"}' >"$DIR/questions-string.json"
expect 2 questions-string.json
printf '{"state":"x","questions":{}}' >"$DIR/no-questions.json"
expect 2 no-questions.json
printf '' >"$DIR/empty.json"
expect 2 empty.json
printf '[1,]' >"$DIR/array-trailing-comma.json"
expect 2 array-trailing-comma.json
printf '{"state":01,%s}' "$Q" >"$DIR/bad-number.json"
expect 2 bad-number.json

exit $fail
