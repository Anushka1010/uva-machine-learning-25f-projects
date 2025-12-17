"use client";

import LayoutShell from "@/components/LayoutShell";
import Card from "@/components/ui/Card";
import Badge from "@/components/ui/Badge";
import {
  getCategoryImage,
  listCategoryImages,
} from "@/lib/categoryImages";

const CLASSES = [
  "asian pear",
  "cucumber",
  "eggs",
  "leafy green",
  "leftovers",
  "orange",
  "sauce",
  "soda",
  "tomato",
];

export default function CategoryImagesPage() {
  const hardcodedImages = listCategoryImages();

  return (
    <LayoutShell>
      <div className="space-y-4">
        <div>
          <h1 className="text-2xl font-semibold">Class Images</h1>
          <p className="text-sm text-gray-600">
            Place image files in the public/images/ folder with the corresponding names.
          </p>
        </div>

        <Card className="p-4 space-y-3">
          <div className="text-sm text-gray-600">
            Images are hardcoded to paths in public/images/. Upload your image files there manually.
          </div>
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {CLASSES.map((cls) => (
            <Card key={cls} className="p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="font-medium">{cls}</div>
                  <div className="text-xs text-gray-600">
                    Hardcoded: {hardcodedImages[cls] ? "Yes" : "No"}
                  </div>
                </div>
              </div>

              <div className="mt-3 flex gap-3 items-center">
                <div className="w-20 h-20 rounded bg-gray-100 border overflow-hidden flex items-center justify-center">
                  {hardcodedImages[cls] ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={hardcodedImages[cls]} alt="" className="w-full h-full object-cover" />
                  ) : (
                    <span className="text-xs text-gray-500">No image</span>
                  )}
                </div>

                <div className="flex-1 space-y-2">
                  <div className="text-xs text-gray-600">
                    File: {hardcodedImages[cls]}
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </LayoutShell>
  );
}
