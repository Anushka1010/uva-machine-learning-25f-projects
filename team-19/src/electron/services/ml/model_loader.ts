import { pipeline, Pipeline } from '@xenova/transformers';

type TaskType = 'feature-extraction' | 'question-answering' | 'zero-shot-classification' | 'text-classification';

export class ModelLoader {
  private static instance: ModelLoader;
  private pipelines: Map<string, Promise<Pipeline>> = new Map();

  private constructor() {
    // Private constructor to enforce singleton
  }

  public static getInstance(): ModelLoader {
    if (!ModelLoader.instance) {
      ModelLoader.instance = new ModelLoader();
    }
    return ModelLoader.instance;
  }

  /* Retrieves a pipeline for a specific task and model.
     Initializes it if it hasn't been loaded yet. */
  public async getPipeline(task: TaskType, modelName: string): Promise<Pipeline> {
    const key = `${task}:${modelName}`;

    if (!this.pipelines.has(key)) {
      console.log(`[ModelLoader] Loading pipeline for task: ${task}, model: ${modelName}`);
      const pipelinePromise = pipeline(task, modelName);
      this.pipelines.set(key, pipelinePromise as any);
      
      // Handle potential errors during loading to remove the promise from cache
      pipelinePromise.catch((err) => {
        console.error(`[ModelLoader] Failed to load pipeline ${key}:`, err);
        this.pipelines.delete(key);
      });
    }

    return this.pipelines.get(key)!;
  }
}

export const modelLoader = ModelLoader.getInstance();
