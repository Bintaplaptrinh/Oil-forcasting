// GDELT GKG-themed topics for the oil-price news pipeline.
// Queries run against the GDELT DOC 2.0 API (indexes the Global Knowledge Graph).
//
// Query rules (from the API):
//   - ONE parenthesized OR-group + operators; adjacent groups `(A)(B)` are rejected.
//   - Keep the OR-group SHORT (~6 phrases max) or GDELT returns "too short or too long".
//   - Multi-word phrases keep results oil-specific.
//   - `theme:` (GKG theme) works standalone, e.g. `theme:ECON_OILPRICE`.

export const GKG_TOPICS = {
  opec: {
    label: 'OPEC / OPEC+',
    gdeltQuery:
      '("OPEC" OR "OPEC+" OR "OPEC production cut" OR "oil output cut" OR ' +
      '"oil cartel" OR "production quota") sourcelang:eng',
  },
  conflict: {
    label: 'Conflict (oil-relevant)',
    gdeltQuery:
      '("Strait of Hormuz" OR "oil pipeline attack" OR "refinery attack" OR ' +
      '"oil tanker attack" OR "Red Sea shipping" OR "oil supply disruption") sourcelang:eng',
  },
  sanctions: {
    label: 'Sanctions (oil-relevant)',
    gdeltQuery:
      '("oil sanctions" OR "crude embargo" OR "oil price cap" OR "Russian oil ban" OR ' +
      '"Iran oil sanctions" OR "energy sanctions") sourcelang:eng',
  },
  oilprice_theme: {
    label: 'GKG theme: oil price',
    gdeltQuery: 'theme:ECON_OILPRICE sourcelang:eng',
  },
};

// OPEC + OPEC+ member countries (short OR-groups, oil-focused).
const _C = (key, name, terms) => ({
  key: 'country_' + key,
  name,
  gdeltQuery: '(' + terms.join(' OR ') + ') sourcelang:eng',
});

export const OPEC_COUNTRIES = [
  // --- OPEC ---
  _C('saudi_arabia', 'Saudi Arabia', ['"Saudi Arabia oil"', '"Saudi Aramco"', '"Saudi crude"']),
  _C('iran',         'Iran',         ['"Iran oil"', '"Iranian oil exports"', '"Iran crude"']),
  _C('iraq',         'Iraq',         ['"Iraq oil"', '"Basra crude"', '"Iraq OPEC"']),
  _C('uae',          'UAE',          ['"UAE oil"', '"ADNOC"', '"Abu Dhabi oil"']),
  _C('kuwait',       'Kuwait',       ['"Kuwait oil"', '"Kuwait Petroleum"']),
  _C('venezuela',    'Venezuela',    ['"Venezuela oil"', '"PDVSA"', '"Venezuelan crude"']),
  _C('nigeria',      'Nigeria',      ['"Nigeria oil"', '"Nigerian crude"']),
  _C('libya',        'Libya',        ['"Libya oil"', '"Libyan crude"']),
  _C('algeria',      'Algeria',      ['"Algeria oil"', '"Sonatrach"']),
  _C('angola',       'Angola',       ['"Angola oil"', '"Angolan crude"']),
  // --- OPEC+ ---
  _C('russia',       'Russia',       ['"Russian oil"', '"Rosneft"', '"Urals crude"']),
  _C('kazakhstan',   'Kazakhstan',   ['"Kazakhstan oil"', '"KazMunayGas"', '"CPC blend"']),
  _C('mexico',       'Mexico',       ['"Mexico oil"', '"Pemex"']),
  _C('oman',         'Oman',         ['"Oman oil"', '"Oman crude"']),
  _C('azerbaijan',   'Azerbaijan',   ['"Azerbaijan oil"', '"SOCAR"', '"BTC pipeline"']),
];
