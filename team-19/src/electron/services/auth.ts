import Store from 'electron-store';

export interface AuthKeys {
  canvasToken: string;
  notionToken: string;
  canvasDomain: string;
  notionDbId: string;
  openaiKey?: string;
}

const store = new Store<AuthKeys>({
  defaults: {
    canvasToken: '',
    notionToken: '',
    canvasDomain: 'canvas.instructure.com',
    notionDbId: '',
    openaiKey: '',
  },
});

export const saveKeys = (keys: AuthKeys): void => {
  store.set('canvasToken', keys.canvasToken);
  store.set('notionToken', keys.notionToken);
  store.set('canvasDomain', keys.canvasDomain);
  store.set('notionDbId', keys.notionDbId);
  if (keys.openaiKey) {
    store.set('openaiKey', keys.openaiKey);
  }
};

export const getKeys = (): AuthKeys => {
  return {
    canvasToken: store.get('canvasToken'),
    notionToken: store.get('notionToken'),
    canvasDomain: store.get('canvasDomain'),
    notionDbId: store.get('notionDbId'),
    openaiKey: store.get('openaiKey'),
  };
};

export const clearKeys = (): void => {
  store.clear();
};
