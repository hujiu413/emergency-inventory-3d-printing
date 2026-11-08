# Sources and access

## EM-DAT historical events

**EM-DAT Public Data (Emergency Events Database), UCLouvain/CRED**:
<https://public.emdat.be/>.

The exact original download date/release was not supplied. The metadata's
2000–2025 range is a screening window; retained event years are 2000–2020.
After obtaining EM-DAT access, retrieve the 27 identifiers in
`data/provenance/event_identifiers.csv`. The `dis_no` column corresponds to
`DisNo.`. The local `sample` column identifies optimization scenarios.

The raw event export and original event-level location/administrative records are
excluded. The complete demand CSV and optimization JSON are excluded as well:
their water-kit totals recover each original Total Affected value. Source users
can prepare these inputs locally with the supplied helper. The public allocation
table records modeled weights, not observed subnational impact counts.

EM-DAT's FAQ permits sharing authors' variables/results alongside event identifiers
but restricts redistribution of Public Data. This repository supplies model
parameters, identifiers, code and aggregate analytical outputs without reproducing
the source event table. It grants no additional rights to EM-DAT data.

- FAQ: <https://doc.emdat.be/docs/additional-resources-and-tutorials/faq/>
- Terms: <https://doc.emdat.be/docs/legal/terms-of-use/>
- Citation policy: <https://doc.emdat.be/docs/legal/citation-policy/>
- Public table: <https://doc.emdat.be/docs/data-structure-and-content/emdat-public-table/>
- Human impacts: <https://doc.emdat.be/docs/data-structure-and-content/impact-variables/human/>

These documentation pages were consulted on 2026-11-08. That is not the original
data download date. The data collector should record the actual download date
when available; it must not be invented.

## Other references recorded in the supplied source material

- Formlabs Form 4B: <https://formlabs.com/3d-printers/form-4b/>
- Formlabs BioMed Clear Resin: <https://formlabs.com/global/products/biomed-clear-resin/>
- Sphere Handbook (2018): <https://spherestandards.org/wp-content/uploads/Sphere-Handbook-2018-EN.pdf>
- UK preparedness context: <https://www.gov.uk/government/publications/pandemic-preparedness-strategy-building-our-capabilities/uk-government-approach-to-implementing-the-strategy-england-only>

These are contextual source references, not certification of every model
coefficient as an observed or current device/market specification. Copies of
third-party publications are not bundled.
