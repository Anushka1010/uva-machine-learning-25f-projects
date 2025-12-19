export type ItemMacros = {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  serving: string;
};

export const DEFAULT_ITEM_MACROS: Record<string, ItemMacros> = {
  "asian pear": { calories: 100, protein: 1, carbs: 26, fat: 0, serving: "1 medium" },
  cucumber: { calories: 16, protein: 1, carbs: 4, fat: 0, serving: "1 cup" },
  eggs: { calories: 155, protein: 13, carbs: 1, fat: 11, serving: "2 large" },
  "leafy green": { calories: 20, protein: 2, carbs: 3, fat: 0, serving: "2 cups" },
  leftovers: { calories: 400, protein: 20, carbs: 40, fat: 15, serving: "1 container" },
  orange: { calories: 62, protein: 1, carbs: 15, fat: 0, serving: "1 medium" },
  sauce: { calories: 60, protein: 1, carbs: 6, fat: 4, serving: "2 tbsp" },
  soda: { calories: 140, protein: 0, carbs: 39, fat: 0, serving: "12 oz" },
  tomato: { calories: 22, protein: 1, carbs: 5, fat: 0, serving: "1 medium" },
};
