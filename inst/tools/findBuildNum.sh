#!/bin/sh

set -o errexit
set -o nounset

if ! which git >/dev/null; then
  echo 1.0
else
  git describe | sed -e 's/^v//' -e 's/-[^-]*$//'
fi
