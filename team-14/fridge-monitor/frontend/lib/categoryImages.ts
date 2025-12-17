// Hardcoded image paths for each class
const hardcodedImages: Record<string, string> = {
  "asian pear": "/images/asian-pear.png",
  "cucumber": "/images/cucumber.png",
  "eggs": "/images/eggs.png",
  "leafy green": "/images/leafy-green.png",
  "leftovers": "/images/leftovers.png",
  "orange": "/images/orange.png",
  "sauce": "/images/sauce.png",
  "soda": "/images/soda.png",
  "tomato": "/images/tomato.png",
};

export async function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.onload = () => resolve(reader.result as string);
    reader.readAsDataURL(file);
  });
}

export function getCategoryImage(labelOrCategory: string): string | null {
  const key = labelOrCategory.toLowerCase();
  return hardcodedImages[key] || null;
}

export function setCategoryImage(labelOrCategory: string, dataUrl: string) {
  // For hardcoding: alert the base64 so user can copy it to the code
  alert(`Copy this base64 to hardcodedImages["${labelOrCategory.toLowerCase()}"] in categoryImages.ts:\n\n${dataUrl}`);
}

export function removeCategoryImage(labelOrCategory: string) {
  // Not applicable for hardcoded images
}

export function listCategoryImages() {
  return hardcodedImages;
}