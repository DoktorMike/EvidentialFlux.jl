firstrelease:
	npx standard-version --first-release

prerelease:
	npx standard-version --prerelease

release:
	#npx standard-version --release-as $VERSION
	npx standard-version

releasemajor:
	perl -pi -e 's/(version = ")(\d+)\.(\d+)\.(\d+)/$1.($2+1).".".$3.".".$4/ge' Project.toml
	npx standard-version -r=major

releaseminor:
	perl -pi -e 's/(version = ")(\d+)\.(\d+)\.(\d+)/$1.$2.".".($3+1).".".$4/ge' Project.toml
	npx standard-version -r=minor

releasepatch:
	perl -pi -e 's/(version = ")(\d+)\.(\d+)\.(\d+)/$1.$2.".".$3.".".($4+1)/ge' Project.toml
	npx standard-version -r=patch

