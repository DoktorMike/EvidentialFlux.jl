# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## [1.9.0](https://github.com/DoktorMike/EvidentialFlux.jl/compare/v1.8.0...v1.9.0) (2026-04-12)


### Features

* added beta binomial and exponential gamma layers for proportions and positive continues responses respectively. ([1f8826d](https://github.com/DoktorMike/EvidentialFlux.jl/commit/1f8826db14d28565a92db777874944fddbc15f2d))


### Bug Fixes

* fixed an inconsistency when predict_mean was actually predicting in parameter space instead of data space. ([89d05f7](https://github.com/DoktorMike/EvidentialFlux.jl/commit/89d05f7b72511dbbd58eaf60542eb4ea06dcc9d0))


### Other

* changed the numbered loss names to something more descriptive {layer}loss_{extension} kind of.. don't judge me, naming is hard. ([1417b5c](https://github.com/DoktorMike/EvidentialFlux.jl/commit/1417b5c7efda0a2b01ab9d383943d5d250295c6e))


### Documentation

* added a section that catches people looking for the bernoulli distribution for normal classification purposes and outlier detection. ([334ab68](https://github.com/DoktorMike/EvidentialFlux.jl/commit/334ab68ed5e5f992f32eace20f5731b01987acea))
* cleaned up documentation and put some more guidance for how to actually use this package, especially on how to deal with the evidence regularization parameters. ([ff911b1](https://github.com/DoktorMike/EvidentialFlux.jl/commit/ff911b153894c296789eca5f936cd7353101927c))
* created a guide for when to use which layer depending on common use cases which is hopefully useful for practitioners. ([ee88346](https://github.com/DoktorMike/EvidentialFlux.jl/commit/ee883460030ac91c70592b6a18191c0dc9064367))
* documented a few parallels regarding the supported distributions we have and how they can be used as special cases for distributions not explicitely supported. ([8a0edbc](https://github.com/DoktorMike/EvidentialFlux.jl/commit/8a0edbc2e02304550ff1bb93ac78eef9608d0643))

## [1.8.0](https://github.com/DoktorMike/EvidentialFlux.jl/compare/v1.7.0...v1.8.0) (2026-04-11)


### Features

* added a consistent api for uncertainty, aleatoric and epistemic. ([539e4f8](https://github.com/DoktorMike/EvidentialFlux.jl/commit/539e4f89cb712ea5766f559015cdd5b991ed9d1e))
* added a new api for prediction of the evidential models. Now predict outputs distributional parameters while predictive outputs the values in data space, i.e., the posterior predictive mean. ([dde33d0](https://github.com/DoktorMike/EvidentialFlux.jl/commit/dde33d0bcb8a1a0a36334371eb766acc146d48b2))
* Added a PoissonGamma (PG) layer for count regression. ([5e92c2e](https://github.com/DoktorMike/EvidentialFlux.jl/commit/5e92c2ef7af36ae7caa113be43fb2a88b573f692))
* Added F-DIR layer which is a flexible Dirichlet distribution for classification purposes. It should be strictly more powerful and also sporting multimodality. ([ad3136c](https://github.com/DoktorMike/EvidentialFlux.jl/commit/ad3136c8963a48078511c5a5a85f728b4426496f))
* Added the Beta-Negative Binomial layer. We're using a marginal negative log likelihood and one featuring and evidence regularizer. ([8d8c1cf](https://github.com/DoktorMike/EvidentialFlux.jl/commit/8d8c1cff5e94e8d860c12cd7961e95ff3b2819f9))
* Added the Multinomial Dirichlet loss function which allows us to fit a count of multiple classes. This reuses the DIR layer so we introduce only a new loss. ([39017ea](https://github.com/DoktorMike/EvidentialFlux.jl/commit/39017ea0631bc8bc2d0f4c0e4bd7a70109c83399))


### Other

* 🧪 more tests confirming the correctness of the new FDIR layer. ([8bb7ca9](https://github.com/DoktorMike/EvidentialFlux.jl/commit/8bb7ca96a14c19aa6b3f70c5241644b5bcf3d17e))

## [1.7.0](https://github.com/DoktorMike/EvidentialFlux.jl/compare/v1.6.0...v1.7.0) (2026-04-11)


### Features

* Added a new dirichlet loss for evidence correction by Pandey et. al. 2025 ([c220d75](https://github.com/DoktorMike/EvidentialFlux.jl/commit/c220d75b39246233559eb60d4d6dcdaf8c63f6d4))


### Documentation

* Rewrite of the README to introduce all use cases clearly. ([1bc831a](https://github.com/DoktorMike/EvidentialFlux.jl/commit/1bc831ab4c1ff5b45e6a55add5d28d47b89b4b77))


### Other

* big refactor adding an abstract evidential layer and building out helper functions. ([62be40f](https://github.com/DoktorMike/EvidentialFlux.jl/commit/62be40f0365a5ddc7f09402302706266da6de321))
* clean up code and add a shitload of tests. ([d9e79b6](https://github.com/DoktorMike/EvidentialFlux.jl/commit/d9e79b68725c656efeabb965f1bb3960080dc223))

## [1.6.0](https://github.com/DoktorMike/EvidentialFlux.jl/compare/v1.5.3...v1.6.0) (2025-11-09)


### Features

* added new losses for regression. ([b0805be](https://github.com/DoktorMike/EvidentialFlux.jl/commit/b0805be43bc5c21c63dcd329593376e75095975a))
* added the uncertainty corrected loss for DER. ([326ca81](https://github.com/DoktorMike/EvidentialFlux.jl/commit/326ca8121e0587ce795929ab618cbd8f91ca1f61))


### Documentation

* 📛 badges addition for new formatter. ([82aec5c](https://github.com/DoktorMike/EvidentialFlux.jl/commit/82aec5c04e8b8c53b43ed3c3d5c840ec03b4f42a))

### [1.5.3](https://github.com/DoktorMike/EvidentialFlux.jl/compare/v1.5.2...v1.5.3) (2025-09-30)


### Bug Fixes

* finally fixed the damn documentation. ([a23251c](https://github.com/DoktorMike/EvidentialFlux.jl/commit/a23251c2844161a59876a6f506ef1916513980a1))


### Documentation

* clarified the predict function for MVE networks. ([4973a4c](https://github.com/DoktorMike/EvidentialFlux.jl/commit/4973a4caa5e79f3caea718f7c7f316ca3a54c6fa))


### Other

* 🦋 formatting according to runic. ([8e8888c](https://github.com/DoktorMike/EvidentialFlux.jl/commit/8e8888c1de60a76824f40c74c8b79207fdc66d17))

### [1.5.2](https://github.com/DoktorMike/EvidentialFlux.jl/compare/v1.5.1...v1.5.2) (2024-06-19)


### Bug Fixes

* fixed a bug in the prediction function of MVE network. ([fcada7d](https://github.com/DoktorMike/EvidentialFlux.jl/commit/fcada7d9f77fea2f776f589a9805c4c080eec825))

### [1.5.1](https://github.com/DoktorMike/EvidentialFlux.jl/compare/v1.5.0...v1.5.1) (2024-06-18)


### Documentation

* fixed the tag. ([4262fa3](https://github.com/DoktorMike/EvidentialFlux.jl/commit/4262fa397fcaeb6047e5d9b0e2f199b111c34a0e))

## [1.5.0](https://github.com/DoktorMike/EvidentialFlux.jl/compare/v1.4.0...v1.5.0) (2024-06-18)


### Features

* Made explicit mu and sigma layers for MVE network. ([97165ab](https://github.com/DoktorMike/EvidentialFlux.jl/commit/97165ab9b021eda833404ae2272d48553b545a58))


### Documentation

* added more docs. ([7784a84](https://github.com/DoktorMike/EvidentialFlux.jl/commit/7784a84ebb489cb13f9ce6770de37e4464059310))
* better workflow. ([91ea4d7](https://github.com/DoktorMike/EvidentialFlux.jl/commit/91ea4d71f4f633ac9262e32885f6f73dfa64fdc2))
* cleaning up documentation and dead links. ([c33b8f8](https://github.com/DoktorMike/EvidentialFlux.jl/commit/c33b8f8f2d0b1546ea3b232ad3424835b2a7b582))
* no idea why this fails. ([f87fa9a](https://github.com/DoktorMike/EvidentialFlux.jl/commit/f87fa9a4791c603ab68f5332de20c432def21c0e))
* redeploying docs. ([7fc8c65](https://github.com/DoktorMike/EvidentialFlux.jl/commit/7fc8c653de007fd875e98c0e9e69b58170ce71de))
* updated documentation job. ([9f6bc9f](https://github.com/DoktorMike/EvidentialFlux.jl/commit/9f6bc9fa426f84afab2a34f796502dc6dfabcf8b))

## [1.4.0](https://github.com/DoktorMike/EvidentialFlux.jl/compare/v1.3.3...v1.4.0) (2024-06-17)


### Features

* Implemented Mean-variance network and cleaned up the dependencies. ([448e865](https://github.com/DoktorMike/EvidentialFlux.jl/commit/448e865900420055573e442702135807eaddefdd))

### [1.3.3](https://github.com/DoktorMike/EvidentialFlux.jl/compare/v1.3.2...v1.3.3) (2023-04-25)


### Other

* Added automatic formatting check for SciML compliance. ([3fe21d8](https://github.com/DoktorMike/EvidentialFlux.jl/commit/3fe21d8a33f19ee8ab60c9def3a0893796fd7008))


### Documentation

* Added a badge for the style compliance check. ([8bdbee8](https://github.com/DoktorMike/EvidentialFlux.jl/commit/8bdbee89b947bc18101879c3ea159260fb316b2f))
* Added a DOI Zenodo badge. ([ea3f035](https://github.com/DoktorMike/EvidentialFlux.jl/commit/ea3f0351703bb1977711e9975040142e0001b8a2))
* small corrections ([708f4b1](https://github.com/DoktorMike/EvidentialFlux.jl/commit/708f4b13c6010db2cc68729781f806b43d926c0c))

### [1.3.2](https://github.com/DoktorMike/EvidentialFlux.jl/compare/v1.3.1...v1.3.2) (2022-07-26)


### Documentation

* Added a favicon as well. ([b516ea5](https://github.com/DoktorMike/EvidentialFlux.jl/commit/b516ea5b5a7588a960a241ca5f9043d449441fb3))
* Made an EvidentialFlux logo based on the julia logo. :) ([5a66ab6](https://github.com/DoktorMike/EvidentialFlux.jl/commit/5a66ab672403171a0e71c26c2d7e92a004255a81))


### Other

* Broke out the negative log likelihood calculation from the losses. ([9e45c90](https://github.com/DoktorMike/EvidentialFlux.jl/commit/9e45c902facf2dfb16b6d8c8af11d85f0bdfbc80))
* Fixing styling inconsistencies with scimlstyle. ([5259cd1](https://github.com/DoktorMike/EvidentialFlux.jl/commit/5259cd1ed8f8ebafdb91ab3a4c3b5ab6fe560d0c))

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
