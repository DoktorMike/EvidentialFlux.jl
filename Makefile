firstrelease:
	npx standard-version --first-release

prerelease:
	npx standard-version --prerelease

release:
	#npx standard-version --release-as $VERSION
	npx standard-version

releasemajor:
	bump.sh major

releaseminor:
	bump.sh minor

releasepatch:
	bump.sh patch

format:
	julia -m Runic --inplace src test

formatcheck:
	julia -m Runic --check --diff src test
