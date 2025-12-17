import type {
  Detection,
  ExpiryDefaults,
  FridgeEvent,
  Item,
  Recipe,
  RecipeRecommendation,
} from "./types";
import { daysFromNowISO, id, macroDistance, nowISO } from "./utils";
import { DEFAULT_ITEM_MACROS, type ItemMacros } from "./nutrition";

/**
 * -------------------------
 * In-memory state (persisted to localStorage)
 * -------------------------
 */
let expiryDefaults: ExpiryDefaults = {
  produce: 5,
  dairy: 7,
  protein: 4,
  grains: 90,
  condiments: 365,
  other: 14,
};

let items: Item[] = [];
let events: FridgeEvent[] = [];

/**
 * -------------------------
 * Persist mock DB in localStorage
 * (Fixes "Item not found" on route changes / refresh)
 * -------------------------
 */
const DB_STORAGE_KEY = "fridge_mock_db_v1";

type StoredDb = {
  items: Item[];
  events: FridgeEvent[];
  expiryDefaults: ExpiryDefaults;
};

let dbLoaded = false;

function loadDb(): void {
  if (dbLoaded) return;
  dbLoaded = true;

  if (typeof window === "undefined") return;

  try {
    const raw = localStorage.getItem(DB_STORAGE_KEY);
    if (!raw) return;
    const parsed: StoredDb = JSON.parse(raw);

    items = parsed.items ?? items;
    events = parsed.events ?? events;
    expiryDefaults = parsed.expiryDefaults ?? expiryDefaults;
  } catch {
    // ignore
  }
}

function saveDb(): void {
  if (typeof window === "undefined") return;

  const payload: StoredDb = { items, events, expiryDefaults };
  localStorage.setItem(DB_STORAGE_KEY, JSON.stringify(payload));
}

/**
 * -------------------------
 * Editable macros per label (persisted separately)
 * -------------------------
 */
const MACRO_STORAGE_KEY = "fridge_macro_overrides_v1";
let macroOverridesCache: Record<string, ItemMacros> | null = null;

function loadMacroOverrides(): Record<string, ItemMacros> {
  if (macroOverridesCache) return macroOverridesCache;

  if (typeof window === "undefined") {
    macroOverridesCache = {};
    return macroOverridesCache;
  }

  try {
    const raw = localStorage.getItem(MACRO_STORAGE_KEY);
    macroOverridesCache = raw ? JSON.parse(raw) : {};
  } catch {
    macroOverridesCache = {};
  }
  return macroOverridesCache!;
}

function saveMacroOverrides(next: Record<string, ItemMacros>) {
  macroOverridesCache = next;
  if (typeof window !== "undefined") {
    localStorage.setItem(MACRO_STORAGE_KEY, JSON.stringify(next));
  }
}

export function api_getItemMacros(label: string): Promise<ItemMacros | null> {
  const key = label.toLowerCase();
  const overrides = loadMacroOverrides();
  return Promise.resolve(overrides[key] ?? DEFAULT_ITEM_MACROS[key] ?? null);
}

export function api_setItemMacros(label: string, macros: ItemMacros): Promise<{ ok: true }> {
  const key = label.toLowerCase();
  const overrides = loadMacroOverrides();
  saveMacroOverrides({ ...overrides, [key]: macros });
  return Promise.resolve({ ok: true });
}

export function api_resetItemMacros(label: string): Promise<{ ok: true }> {
  const key = label.toLowerCase();
  const overrides = loadMacroOverrides();
  const { [key]: _, ...rest } = overrides;
  saveMacroOverrides(rest);
  return Promise.resolve({ ok: true });
}

/**
 * -------------------------
 * Recipes (ONLY from your 9 classes)
 * -------------------------
 */
const recipes: Recipe[] = [
  {
    id: "r1",
    name: "Cucumber Tomato Salad",
    ingredients: ["cucumber", "tomato", "leafy green", "sauce"],
    macros: { calories: 220, protein: 6, carbs: 18, fat: 14 },
    steps: ["Chop cucumber and tomato.", "Toss with leafy green and sauce."],
  },
  {
    id: "r2",
    name: "Tomato Egg Scramble",
    ingredients: ["eggs", "tomato", "sauce"],
    macros: { calories: 340, protein: 22, carbs: 10, fat: 22 },
    steps: ["Whisk eggs.", "Cook tomato briefly, add eggs.", "Finish with sauce."],
  },
  {
    id: "r3",
    name: "Leftovers Bowl",
    ingredients: ["leftovers", "leafy green", "sauce"],
    macros: { calories: 500, protein: 25, carbs: 45, fat: 20 },
    steps: ["Warm leftovers.", "Serve over leafy green.", "Add sauce."],
  },
  {
    id: "r4",
    name: "Fruit Plate",
    ingredients: ["asian pear", "orange"],
    macros: { calories: 180, protein: 2, carbs: 46, fat: 0 },
    steps: ["Slice fruit and serve."],
  },
  {
    id: "r5",
    name: "Orange Soda Spritzer",
    ingredients: ["orange", "soda"],
    macros: { calories: 140, protein: 1, carbs: 34, fat: 0 },
    steps: ["Pour soda over ice.", "Add orange slices."],
  },
];

/**
 * -------------------------
 * Mock detection labels (ONLY your 9 classes)
 * -------------------------
 */
const LABELS: Array<[string, string]> = [
  ["asian pear", "produce"],
  ["cucumber", "produce"],
  ["eggs", "protein"],
  ["leafy green", "produce"],
  ["leftovers", "other"],
  ["orange", "produce"],
  ["sauce", "condiments"],
  ["soda", "beverage"],
  ["tomato", "produce"],
];

function hashStr(s: string) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h;
}

export function mockInfer(fileName: string): Detection[] {
  // If filename is exactly a known label, return that detection
  const exact = LABELS.find(([label]) => label === fileName.toLowerCase());
  if (exact) {
    return [{
      label: exact[0],
      category: exact[1],
      confidence: 0.95,
      bbox: [0.1, 0.2, 0.2, 0.3],
    }];
  }

  const h = hashStr(fileName || "image");
  const picks = [h % LABELS.length, (h + 7) % LABELS.length, (h + 19) % LABELS.length];

  const seen = new Set<string>();
  const out: Detection[] = [];

  for (let i = 0; i < picks.length; i++) {
    const [label, category] = LABELS[picks[i]];
    if (seen.has(label)) continue;
    seen.add(label);
    out.push({
      label,
      category,
      confidence: Number((0.78 + 0.06 * (i % 3)).toFixed(2)),
      bbox: [0.1 + i * 0.2, 0.2, 0.2, 0.3],
    });
  }

  return out;
}

/**
 * -------------------------
 * "API" functions (frontend-only)
 * -------------------------
 */
export function api_getItem(itemId: string) {
  loadDb();
  const it = items.find((x) => x.id === itemId) ?? null;
  return Promise.resolve(it);
}

export function api_listItems(status?: string) {
  const FLASK_API_URL = "http://127.0.0.1:5000";
  console.log('Fetching items from backend');
  return fetch(`${FLASK_API_URL}/api/inventory`)
    .then((res) => {
      console.log('Fetch response:', res);
      return res.json();
    })
    .then((data) => {
      console.log('Fetched data:', data);
      // Transform backend data to match frontend types
      const transformed = data.map((item: any) => ({
        id: item._id,
        label: item.name,
        category: "food", // or map from name
        expiresAt: item.expiration_date,
        status: item.status,
        imageUrl: item.image?.url,
        datePlaced: item.date_placed,
      }));
      console.log('Transformed:', transformed);
      // Filter by status if provided
      const filtered = status ? transformed.filter((i) => i.status === status) : transformed;
      console.log('Filtered:', filtered);
      return filtered;
    })
    .catch((error) => {
      console.error("Failed to fetch inventory:", error);
      return [];
    });
}

export function api_listEvents() {
  const FLASK_API_URL = "http://127.0.0.1:5000";
  return fetch(`${FLASK_API_URL}/api/events`)
    .then((res) => res.json())
    .catch((error) => {
      console.error("Failed to fetch events:", error);
      return [];
    });
}

export function api_getExpiryDefaults() {
  loadDb();
  return Promise.resolve({ ...expiryDefaults });
}

export function api_setExpiryDefaults(next: ExpiryDefaults) {
  loadDb();
  expiryDefaults = { ...next };
  saveDb();
  return Promise.resolve({ ok: true });
}

export function api_patchItem(itemId: string, patch: Partial<Item>) {
  loadDb();
  items = items.map((it) => (it.id === itemId ? { ...it, ...patch } : it));
  saveDb();
  return Promise.resolve(items.find((x) => x.id === itemId)!);
}

export function api_scanIn(file: { name: string; imageUrl?: string }) {
  loadDb();

  const detections = mockInfer(file.name);
  const createdAt = nowISO();

  const created: Item[] = detections.map((d) => {
    const expDays = (expiryDefaults as any)[d.category] ?? expiryDefaults.other ?? 14;
    return {
      id: id(),
      label: d.label,
      category: d.category,
      confidence: d.confidence,
      createdAt,
      expiresAt: daysFromNowISO(expDays),
      status: "in_fridge",
    };
  });

  items = [...created, ...items];

  const ev: FridgeEvent = {
    id: id(),
    type: "IN",
    timestamp: createdAt,
    imageUrl: file.imageUrl,
    detections,
    resultSummary: `Added ${created.length} item(s)`,
  };

  events = [ev, ...events];

  saveDb();
  return Promise.resolve({ event: ev, createdItems: created });
}

export function api_scanOut(file: { name: string; imageUrl?: string }) {
  loadDb();

  const detections = mockInfer(file.name);
  const ts = nowISO();

  const target = detections[0]?.label;
  let removed: Item | undefined;

  if (target) removed = items.find((i) => i.status === "in_fridge" && i.label === target);
  if (!removed) removed = items.find((i) => i.status === "in_fridge");

  if (!removed) {
    const ev: FridgeEvent = {
      id: id(),
      type: "OUT",
      timestamp: ts,
      imageUrl: file.imageUrl,
      detections,
      resultSummary: "No items to remove",
    };
    events = [ev, ...events];
    saveDb();
    return Promise.resolve({ event: ev, removedItem: null });
  }

  items = items.map((it) =>
    it.id === removed!.id
      ? { ...it, status: "removed" }
      : it
  );

  const ev: FridgeEvent = {
    id: id(),
    type: "OUT",
    timestamp: ts,
    imageUrl: file.imageUrl,
    detections,
    resultSummary: `Removed 1 item: ${removed.label}`,
  };

  events = [ev, ...events];

  saveDb();
  return Promise.resolve({ event: ev, removedItem: { ...removed, status: "removed" as const } });
}

export function api_alerts() {
  const FLASK_API_URL = "http://127.0.0.1:5000";
  return fetch(`${FLASK_API_URL}/api/alerts`)
    .then((res) => res.json())
    .catch((error) => {
      console.error("Failed to fetch alerts:", error);
      return [];
    });
}

export function api_relatedRecipesForItem(label: string) {
  loadDb();

  const ingredient = label.toLowerCase();
  const inv = items.filter((i) => i.status === "in_fridge");
  const invSet = new Set(inv.map((i) => i.label.toLowerCase()));

  const related = recipes
    .filter((r) => r.ingredients.map((x) => x.toLowerCase()).includes(ingredient))
    .map((r) => {
      const have = r.ingredients.filter((ing) => invSet.has(ing.toLowerCase()));
      const missing = r.ingredients.filter((ing) => !invSet.has(ing.toLowerCase()));
      const coverage = have.length / Math.max(1, r.ingredients.length);
      return { recipe: r, have, missing, coverage };
    })
    .sort((a, b) => b.coverage - a.coverage);

  return Promise.resolve(related);
}

export function api_recommendations(payload: {
  target: { calories: number; protein: number; carbs: number; fat: number };
  minCoverage: number;
  prioritizeExpiring: boolean;
}) {
  const FLASK_API_URL = "http://127.0.0.1:5000";
  return fetch(`${FLASK_API_URL}/api/inventory`)
    .then((res) => res.json())
    .then((data) => {
      // Transform to match mock Item format
      const inv = data
        .filter((item: any) => item.status === "in_fridge")
        .map((item: any) => ({
          id: item._id,
          label: item.name,
          category: "food",
          expiresAt: item.expiration_date,
          status: item.status,
        }));

      const invBy = new Map(inv.map((i) => [i.label, i]));
      const now = new Date();

      const scored: RecipeRecommendation[] = recipes.map((r) => {
        const have: string[] = [];
        const missing: string[] = [];
        let expBonus = 0;

        for (const ing of r.ingredients) {
          const it = invBy.get(ing);
          if (it) {
            have.push(ing);
            if (payload.prioritizeExpiring && it.expiresAt) {
              const daysLeft = (new Date(it.expiresAt).getTime() - now.getTime()) / 86400000;
              if (daysLeft <= 2) expBonus += 3;
              else if (daysLeft <= 5) expBonus += 2;
              else expBonus += 1;
            }
          } else {
            missing.push(ing);
          }
        }

        const coverage = have.length / Math.max(1, r.ingredients.length);
        const base = coverage * 10 + expBonus - missing.length * 1.5;

        const md = macroDistance(r.macros, payload.target);
        const score = base - 0.02 * md;

        return {
          recipe: r,
          score: Number(score.toFixed(2)),
          coverage,
          have,
          missing,
          reasons: coverage >= payload.minCoverage ? ["Good coverage"] : ["Low coverage"],
        };
      });

      return scored
        .filter((r) => r.coverage >= payload.minCoverage)
        .sort((a, b) => b.score - a.score)
        .slice(0, 10);
    })
    .catch((error) => {
      console.error("Failed to fetch recommendations:", error);
      return [];
    });
}

export function api_listRecipes() {
  loadDb();
  return Promise.resolve([...recipes]);
}
