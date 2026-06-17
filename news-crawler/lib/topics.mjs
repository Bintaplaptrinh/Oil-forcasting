// Topic definitions for the three target themes.
// - `gdeltQuery` : a GDELT DOC 2.0 boolean query (English only).
// - `keywords`   : lowercase terms used to classify free-text headlines
//                  (used by the oilprice.com slug filter and for re-tagging).
//
// A headline matches a topic if ANY keyword is found as a word/substring.

export const TOPICS = {
  war: {
    label: 'War / Conflict',
    gdeltQuery:
      '(war OR conflict OR military OR invasion OR missile OR airstrike OR ceasefire OR insurgency OR warship OR offensive) sourcelang:eng',
    keywords: [
      'war', 'warfare', 'conflict', 'military', 'troops', 'invasion', 'invade',
      'missile', 'airstrike', 'air strike', 'ceasefire', 'insurgency', 'insurgent',
      'warship', 'offensive', 'combat', 'drone strike', 'rebels', 'militant',
      'army', 'navy', 'clashes', 'shelling', 'bombing', 'frontline', 'siege',
      'hostilities', 'armed', 'coup', 'blockade', 'hormuz',
    ],
  },
  political_economy: {
    label: 'Political Economy',
    gdeltQuery:
      '(sanctions OR tariff OR embargo OR OPEC OR inflation OR "interest rate" OR "central bank" OR recession OR "trade war" OR "price cap") sourcelang:eng',
    keywords: [
      'sanction', 'tariff', 'embargo', 'opec', 'inflation', 'interest rate',
      'central bank', 'recession', 'trade war', 'price cap', 'export ban',
      'import ban', 'subsidy', 'nationalize', 'nationalise', 'gdp', 'stimulus',
      'currency', 'devaluation', 'debt', 'fiscal', 'monetary', 'quota',
      'production cut', 'output cut', 'windfall tax', 'price war', 'demand',
      'supply glut', 'stockpile', 'reserves', 'budget',
    ],
  },
  natural_disaster: {
    label: 'Natural Disaster',
    gdeltQuery:
      '(earthquake OR hurricane OR flood OR wildfire OR drought OR tsunami OR cyclone OR typhoon OR volcano OR landslide) sourcelang:eng',
    keywords: [
      'earthquake', 'hurricane', 'flood', 'flooding', 'wildfire', 'drought',
      'tsunami', 'cyclone', 'typhoon', 'volcano', 'volcanic', 'landslide',
      'storm', 'heatwave', 'heat wave', 'blizzard', 'tornado', 'mudslide',
      'natural disaster', 'severe weather', 'flash flood', 'tropical storm',
      'eruption', 'quake', 'famine',
    ],
  },
};

export const TOPIC_KEYS = Object.keys(TOPICS);

// Classify an arbitrary text against all topics. Returns an array of topic keys.
export function classify(text) {
  const t = (text || '').toLowerCase();
  const hits = [];
  for (const [key, def] of Object.entries(TOPICS)) {
    if (def.keywords.some((k) => t.includes(k))) hits.push(key);
  }
  return hits;
}
