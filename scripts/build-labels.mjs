// Generates public/labels/international-labels.json from the model's own id2label
// (scripts/data/international-labels.raw.json, copied from the upstream config.json).
// Usage: node scripts/build-labels.mjs
import fs from 'node:fs';

const raw = JSON.parse(fs.readFileSync(new URL('./data/international-labels.raw.json', import.meta.url), 'utf8'));

// Country of the manufacturer's origin (brand home country), for the makes present in Stanford Cars.
const MAKES = {
  'Acura': 'Japan', 'AM General': 'USA', 'Aston Martin': 'United Kingdom', 'Audi': 'Germany',
  'Bentley': 'United Kingdom', 'BMW': 'Germany', 'Bugatti': 'France', 'Buick': 'USA', 'Cadillac': 'USA',
  'Chevrolet': 'USA', 'Chrysler': 'USA', 'Daewoo': 'South Korea', 'Dodge': 'USA', 'Eagle': 'USA',
  'Ferrari': 'Italy', 'FIAT': 'Italy', 'Fisker': 'USA', 'Ford': 'USA', 'Geo': 'USA', 'GMC': 'USA',
  'Honda': 'Japan', 'HUMMER': 'USA', 'Hyundai': 'South Korea', 'Infiniti': 'Japan', 'Isuzu': 'Japan',
  'Jaguar': 'United Kingdom', 'Jeep': 'USA', 'Lamborghini': 'Italy', 'Land Rover': 'United Kingdom',
  'Lincoln': 'USA', 'Maybach': 'Germany', 'Mazda': 'Japan', 'McLaren': 'United Kingdom',
  'Mercedes-Benz': 'Germany', 'MINI': 'United Kingdom', 'Mitsubishi': 'Japan', 'Nissan': 'Japan',
  'Plymouth': 'USA', 'Pontiac': 'USA', 'Porsche': 'Germany', 'Ram': 'USA', 'Rolls-Royce': 'United Kingdom',
  'Scion': 'Japan', 'smart': 'Germany', 'Spyker': 'Netherlands', 'Suzuki': 'Japan', 'Tesla': 'USA',
  'Toyota': 'Japan', 'Volkswagen': 'Germany', 'Volvo': 'Sweden',
};

// Body-style suffixes -> broad vehicle category.
const BODY = [
  ['Crew Cab', 'Pickup'], ['Extended Cab', 'Pickup'], ['Regular Cab', 'Pickup'], ['Quad Cab', 'Pickup'],
  ['Club Cab', 'Pickup'], ['SuperCab', 'Pickup'], ['Cargo Van', 'Van'], ['Minivan', 'Minivan'],
  ['Van', 'Van'], ['Sedan', 'Sedan'], ['Coupe', 'Coupe'], ['Convertible', 'Convertible'],
  ['Hatchback', 'Hatchback'], ['Wagon', 'Wagon'], ['SUV', 'SUV'],
];

const ids = Object.keys(raw).map(Number).sort((a, b) => a - b);
const out = ids.map((id) => {
  const name = raw[id];
  const m = name.match(/^(.*) (\d{4})$/);
  let rest = m ? m[1] : name;
  const year = m ? Number(m[2]) : null;
  const make = Object.keys(MAKES).find((k) => rest === k || rest.startsWith(k + ' '));
  if (!make) throw new Error('Unknown make in label: ' + name);
  rest = rest.slice(make.length).trim();
  let category = null;
  let body = null;
  for (const [suffix, cat] of BODY) {
    if (rest.endsWith(' ' + suffix) || rest === suffix) {
      body = suffix;
      category = cat;
      rest = rest.slice(0, rest.length - suffix.length).trim();
      break;
    }
  }
  return { id, raw: name, manufacturer: make, model: rest || body || name, body, category, country: MAKES[make], referenceYear: year };
});

fs.mkdirSync(new URL('../public/labels/', import.meta.url), { recursive: true });
fs.writeFileSync(new URL('../public/labels/international-labels.json', import.meta.url), JSON.stringify(out));
console.log('international labels:', out.length);
