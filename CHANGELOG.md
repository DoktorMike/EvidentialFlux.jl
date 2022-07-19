# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## [1.3.0](https://github.com/DoktorMike/EvidentialFlux.jl/compare/v1.2.0...v1.3.0) (2022-07-19)


### Features

* Implemented the DER correction 🥶 ([450a76f](https://github.com/DoktorMike/EvidentialFlux.jl/commit/450a76fc6cd8a55e916a5e4dbc4f791ad7e5efd6)), closes [#9](https://github.com/DoktorMike/EvidentialFlux.jl/issues/9)
* Updated regression example to produce something more useful and more inline with the paper. ([8577d8e](https://github.com/DoktorMike/EvidentialFlux.jl/commit/8577d8e0a91f733ddda477d5877bb050a839182a))


### Documentation

* Added classification image to readme. ([9b3f64f](https://github.com/DoktorMike/EvidentialFlux.jl/commit/9b3f64fd71f0d49acbd3fdd1ca29271111275b24))
* Added documentation of the dirloss function. ([e412f92](https://github.com/DoktorMike/EvidentialFlux.jl/commit/e412f92295c1f2acee2cae51a0e49ea855d6ab5d))
* Added more text regarding the classification entry. ([afb03d1](https://github.com/DoktorMike/EvidentialFlux.jl/commit/afb03d15430806e86819c15a0e14321d934c58dc))
* Added the documentation of the DIR layer. ([736ea62](https://github.com/DoktorMike/EvidentialFlux.jl/commit/736ea623910f5904b9e5d711df7f8456d54a53de))
* Added the regression case to the README. ([5fb3060](https://github.com/DoktorMike/EvidentialFlux.jl/commit/5fb30608d00fe2155cee4b29853be032e61ca2f7))
* Documented the Deep evidential classification. ([f33f872](https://github.com/DoktorMike/EvidentialFlux.jl/commit/f33f87215a45c3e6edcd528171f0ccda6a738c2d))
* Updated the example for the second regression case to use a power of 2 in the loss. ([ec2026b](https://github.com/DoktorMike/EvidentialFlux.jl/commit/ec2026b8ad87b70a5d035e416285ec6fb73771f3))

## [1.2.0](https://github.com/DoktorMike/EvidentialFlux.jl/compare/v1.1.1...v1.2.0) (2022-07-12)


### Features

* Added the MultinomialDirichlet evidential distribution. ([7436f16](https://github.com/DoktorMike/EvidentialFlux.jl/commit/7436f16413d1a0806049b93cad9a0a4586c542ed))


### Documentation

* Made some comments in the example. ([e1f7d08](https://github.com/DoktorMike/EvidentialFlux.jl/commit/e1f7d083bed8ecdc82930737e75479f1e00512ea))
* Updated badges in README. ([e2d9090](https://github.com/DoktorMike/EvidentialFlux.jl/commit/e2d9090c8ca5f27124b97e0238e8e5e6242cf4ea))
* Updated examples for classification. ([daa32f4](https://github.com/DoktorMike/EvidentialFlux.jl/commit/daa32f4a2cfa6bb23e2bad55d9c23ea79c6f0006))

### [1.1.1](https://github.com/DoktorMike/EvidentialFlux/compare/v1.1.0...v1.1.1) (2022-07-09)


### Features

* Added the evidence 🎓 function for the Deep Evidential Regression case. ([5f92f07](https://github.com/DoktorMike/EvidentialFlux/commit/5f92f07ea70d5f62efb5abb2b07b4ff5f88751fa))


### Documentation

* Added documentation workflow to repo. ([63051bd](https://github.com/DoktorMike/EvidentialFlux/commit/63051bd6117a6d26973439141b7330989df4890b))
* Added some information in the README again. ([269fa89](https://github.com/DoktorMike/EvidentialFlux/commit/269fa89edf08e6f660c4d4c41e5d2c8f3cd63be8))
* Added some information in the README again. ([50bae43](https://github.com/DoktorMike/EvidentialFlux/commit/50bae4399d921f76151323071f223dc3401115ff))
* Fixed bug in badges. ([3597864](https://github.com/DoktorMike/EvidentialFlux/commit/35978640936bf00bcc669e5a7b70840756a8e713))
* Rearranged order of the functions. ([da95c62](https://github.com/DoktorMike/EvidentialFlux/commit/da95c62381b64fe8b53223c4f0d15daee54cbc90))
* Section about the foundations of DER. ([fee5f23](https://github.com/DoktorMike/EvidentialFlux/commit/fee5f23d5a898222c4376afe74bcadafb5f46a74))
* Updated README with badge for documentation. ([2ecfbd9](https://github.com/DoktorMike/EvidentialFlux/commit/2ecfbd9b107c67bb38354ee51794f003ea4d9c8a))
* Updated README with more badges. ([e9a5614](https://github.com/DoktorMike/EvidentialFlux/commit/e9a5614de5eae2850e77c0684938fcac11f00b4b))

## 1.1.0 (2022-07-07)


### Features

* Added a full mini example evidential regression case. ([9b3a649](https://github.com/DoktorMike/EvidentialFlux/commit/9b3a649a00e70347cdc5695a849a77ea38d5a1e4))
* Added a predict function that dispatches on the last layer of a Flux Chain. ([f81861a](https://github.com/DoktorMike/EvidentialFlux/commit/f81861a15ad8ef5621f9c15a7fa452f58bf1587a))
* Added calculation function for uncertainty given a NIG network output. ([829937c](https://github.com/DoktorMike/EvidentialFlux/commit/829937c0fb9048650133de40e3e2a5267c9dba5e))
* Added several tests for correctness. ([b4f057d](https://github.com/DoktorMike/EvidentialFlux/commit/b4f057dfba2415884cc95e0336c94dcea3c743a2))
* Added the NormalInverseGamma Dense layer and some tests. ([c245be7](https://github.com/DoktorMike/EvidentialFlux/commit/c245be74f97baa3eb1b1bd99d97945ad47328103))
* Finished the evidential regression loss function. ([3733528](https://github.com/DoktorMike/EvidentialFlux/commit/3733528a615c45669b4b41c9e6f45fd629e1c107))
* Speed up package by removing unneccesary dependencies. ([d1e4631](https://github.com/DoktorMike/EvidentialFlux/commit/d1e4631542385604a2fc6f172962905098768472))
* Updated the tests to include the full evidential regression loss. ([76b9108](https://github.com/DoktorMike/EvidentialFlux/commit/76b91088e315855ed34fe64348ad72266786e9cb))


### Documentation

* Added deployment of docs which is not working yet. ([f3577d3](https://github.com/DoktorMike/EvidentialFlux/commit/f3577d34872a111f22b881a28b6daac2a0260d0b))
* Added documentation for the uncertainty functions ([b5a2dc6](https://github.com/DoktorMike/EvidentialFlux/commit/b5a2dc67a09032cd7e4342c0292e3894c7772681))
* Added Documenter infrastructure. ([437ae08](https://github.com/DoktorMike/EvidentialFlux/commit/437ae085662741d3f572ac922ae8ebd0ee95f473))
* Added load path for local development. ([2765605](https://github.com/DoktorMike/EvidentialFlux/commit/2765605931803b1a20da5007ccc42baa223c39cd))
* Documented the NIG loss functions as well as the aleatoric and epistemic uncertainty. ([5f857e1](https://github.com/DoktorMike/EvidentialFlux/commit/5f857e17ae5f1195ec6ed65494b1cd430e9a8129))
* Finished the documentation such that it can be generated. ([6116f1b](https://github.com/DoktorMike/EvidentialFlux/commit/6116f1ba5b2d15a9ffcf708caff225345be9e35e))
* More documentation regardin the NIG distribution. ([8282c91](https://github.com/DoktorMike/EvidentialFlux/commit/8282c918f780af8623a3f414eb80da7a75aad4cc))
