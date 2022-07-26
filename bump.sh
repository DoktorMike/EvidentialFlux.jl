#!/bin/bash

case "$1" in
	"")
		echo you need to supply major, minor or patch
		exit 1;;
	"major")
		echo Releasing a new Major version
		perl -pi -e 's/(version = ")(\d+)\.(\d+)\.(\d+)/$1.($2+1).".".$3.".".$4/ge' Project.toml
		npx standard-version -r=major;;
	"minor")
		echo "Releasing a new Minor version"
		perl -pi -e 's/(version = ")(\d+)\.(\d+)\.(\d+)/$1.$2.".".($3+1).".".$4/ge' Project.toml
		npx standard-version -r=minor;;
	"patch")
		echo "Releasing a new Patch version"
		perl -pi -e 's/(version = ")(\d+)\.(\d+)\.(\d+)/$1.$2.".".$3.".".($4+1)/ge' Project.toml
		npx standard-version -r=patch;;
	*)
		echo "Dude, something is _off_!"
		exit 1;;
esac

