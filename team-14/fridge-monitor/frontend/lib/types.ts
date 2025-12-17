export type ItemStatus = "in_fridge" | "removed";

export interface Item {
  _id: string;          
  name: string;
  image?: {                
    url: string;
    file_id: string;
  };
  date_placed: string;   
  expiration_date: string; 
  category?: string;
  confidence?: number;
  status?: string;
}

export type FridgeEventType = "IN" | "OUT";

export type FridgeEvent = {
  id: string;
  type: FridgeEventType;
  timestamp: string; // ISO
  imageUrl?: string;
  detections: Detection[];
  resultSummary: string;
};

export type Detection = {
  label: string;
  category: string;
  confidence: number;
  bbox?: [number, number, number, number]; // optional
};

export type Recipe = {
  id: string;
  name: string;
  ingredients: string[];
  macros: { calories: number; protein: number; carbs: number; fat: number };
  steps: string[];
};

export type RecipeRecommendation = {
  recipe: Recipe;
  score: number;
  coverage: number; // 0..1
  have: string[];
  missing: string[];
  reasons: string[];
};

export type ExpiryDefaults = Record<string, number>;
