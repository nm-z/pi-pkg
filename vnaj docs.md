Here is the fully formatted **Markdown** version of the vna/J 3.x user guide for the headless application:

---

# vna/J 3.x

## User Guide for Headless Application

**Author:** Dietmar Krause, DL2SBA
**Address:** Hindenburgstraße 29, D-70794 Filderstadt
**License:** [Creative Commons BY-NC-ND 3.0](http://creativecommons.org/licenses/by-nc-nd/3.0)
**Date:** Saturday, 10 March 2018

---

## Table of Contents

1. [Changes](#changes)
2. [Acknowledgements](#acknowledgements)
3. [Overview](#overview)
4. [History](#history)
5. [Basics](#basics)
6. [Configuration](#configuration)
7. [Command-line Parameters](#command-line-parameters)
8. [Supported Region and Language Codes](#supported-region-and-language-codes)
9. [JAVA Not Found](#java-not-found)
10. [Links](#links)
11. [License](#license)

---

## Changes

| Version | Date             | Who    | Changes                                                                                                |
| ------- | ---------------- | ------ | ------------------------------------------------------------------------------------------------------ |
| 2.9.0   | 26 April 2014    | DL2SBA | Created                                                                                                |
| 3.x     | 13 December 2014 | DL2SBA | Hints & Tips extended                                                                                  |
|         | 4 June 2017      | DL2SBA | Some bugs corrected                                                                                    |
| 3.1     | 9 March 2018     | DL2SBA | New parameters added; average now available via parameter; config file from GUI app no longer relevant |

---

## Acknowledgements

* Monika, DL6SCF
* Davide, IW3HEV and team
* Andy, G0POY
* Dan, AC6LA
* Tamas, HG1DFB
* Erik, SM3HEW
* Erik, OZ4KK
* Bertil, SM6ENG
* Domingo, EA1DAX
* Toshiyuki Urakami, JP1PZE
* Detlef, DL7IY
* Gerrit, PA3DJY
* Many worldwide users
* Ina the cat 🐱

---

## Overview

The **miniVNA** and **miniVNApro** from [Mini Radio Solutions](http://www.miniradiosolutions.com) are compact network analyzers operated via PC software.
Originally Windows-focused, support for Linux was dropped due to compatibility issues.

To address this, starting in **2007**, a cross-platform Java-based application was developed, compatible with:

* Windows (98, XP, 7, Vista, 8, 10 – 32 & 64-bit)
* macOS (32 & 64-bit)

---

## History

The **vna/J GUI** was released in 2007.
Due to user demand for automation, a **headless (command-line) version** was introduced in version **2.9** (April 2014).

---

## Basics

Since version **2.9**, a **headless** mode allows:

* Automated scans
* No GUI required
* Output formats:

  * CSV
  * XLS
  * XML
  * SnP (S-parameters)
  * ZPLOTS

---

## Configuration

The **GUI config file is not used**.
All parameters are passed via **command-line arguments**.

---

## Command-line Parameters

Parameters are passed with `-D` options:

```bash
java -Dfstart=1000000 \
-Dfstop=30000000 \
-Dfsteps=500 \
-DdriverId=1 \
-DdriverPort=COM4 \
-Daverage=1 \
-Dcalfile=TRAN_miniVNA.cal \
-Dscanmode=TRAN \
-Dexports=csv,snp,xml,xls,zplots \
-DexportDirectory="C:\Users\dietmar\vnaJ.3.1\export" \
-DexportFilename="VNA_{0,date,yyMMdd}_{0,time,HHmmss}" \
-DkeepGeneratorOn \
-Duser.home=c:/temp \
-Duser.language=en \
-Duser.region=US \
-jar vnaJ-hl.3.1.21
```

### Supported Parameters

| Parameter         | Mandatory | Description                                   |
| ----------------- | --------- | --------------------------------------------- |
| `user.home`       | No        | Root dir for vna/J                            |
| `user.language`   | No        | Language code                                 |
| `user.region`     | No        | Region code                                   |
| `fstart`          | Yes       | Start frequency in Hz                         |
| `fstop`           | Yes       | Stop frequency in Hz                          |
| `fsteps`          | Yes       | Number of steps                               |
| `calfile`         | Yes       | Full path to calibration file                 |
| `driverId`        | Yes       | Instrument ID (see below)                     |
| `comport`         | Yes       | Serial port (e.g., COM3)                      |
| `average`         | No        | Scan averaging (default = 1)                  |
| `exportDirectory` | Yes       | Output folder                                 |
| `exportFilename`  | Yes       | Output filename pattern                       |
| `scanmode`        | Yes       | `REFL` (reflection) or `TRAN` (transmission)  |
| `exports`         | No        | Formats: `csv`, `snp`, `xml`, `xls`, `zplots` |
| `keepGeneratorOn` | No        | Keeps generator on after scan                 |

**Note:** Parameters are case-sensitive.

### Driver ID Mapping

| ID | Device                |
| -- | --------------------- |
| 0  | Sample                |
| 1  | miniVNA               |
| 2  | miniVNApro            |
| 3  | miniVNApro + Extender |
| 4  | MAX6                  |
| 5  | MAX6-500MHz           |
| 10 | miniVNA LF            |
| 12 | miniVNApro LG         |
| 20 | miniVNAtiny           |
| 30 | MetroVNA              |
| 40 | VNAArduino            |

---

## Supported Region and Language Codes

| Region Code | Language Code | Language  |
| ----------- | ------------- | --------- |
| US          | en            | English   |
| DE          | de            | German    |
| HU          | hu            | Hungarian |
| PL          | pl            | Polish    |
| SE          | sv            | Swedish   |
| IT          | it            | Italian   |
| ES          | es            | Spanish   |
| NL          | nl            | Dutch     |
| CZ          | cs            | Czech     |
| FR          | fr            | French    |
| JP          | ja            | Japanese  |
| RUS         | ru            | Russian   |

---

## JAVA Not Found

On Windows, if Java isn't available on the PATH:

* Edit the `.cmd` script to include the **full path** to your `java.exe`.
* Enclose the path in **quotation marks** if it includes spaces.

---

## Links

* [vna/J homepage](http://vnaj.dl2sba.com)
* [Yahoo group](http://groups.yahoo.com/group/analyzer_iw3hev)
  *(SUSE/Ubuntu install guides in Files > Subjects - Off Topic - Brainstorming)*
* [Mini Radio Solutions](http://www.miniradiosolutions.com)

---

## License

### Dutch

Licensed under [CC BY-NC-ND 3.0 NL](http://creativecommons.org/licenses/by-nc-nd/3.0/nl/)

### English

Licensed under [CC BY-NC-ND 3.0](http://creativecommons.org/licenses/by-nc-nd/3.0/)

### Deutsch

Lizenziert unter [CC BY-NC-ND 3.0 DE](http://creativecommons.org/licenses/by-nc-nd/3.0/de/)

---

Let me know if you’d like this as a downloadable `.md` file!
