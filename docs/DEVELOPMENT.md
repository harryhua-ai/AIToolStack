# Development Guide

This guide covers development practices, patterns, and workflows for AIToolStack.

## Project Structure

```
AIToolStack/
├── backend/                    # FastAPI backend
│   ├── api/                   # API route handlers
│   │   └── routes.py         # All API endpoints
│   ├── services/              # Business logic
│   │   └── training_service.py
│   ├── utils/                 # Utilities
│   │   └── yolo_export.py
│   ├── config.py             # Configuration
│   ├── main.py               # Application entry point
│   └── requirements.txt       # Python dependencies
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── ModelSpace.tsx   # Main model management UI
│   │   │   ├── TrainingPanel.tsx
│   │   │   └── *.tsx         # Other components
│   │   ├── i18n/             # Internationalization
│   │   │   └── locales/      # Translation files
│   │   │       ├── en.json   # English
│   │   │       └── zh.json   # Chinese
│   │   ├── hooks/            # Custom React hooks
│   │   ├── ui/               # UI component library
│   │   │   ├── Button.tsx
│   │   │   ├── Input.tsx
│   │   │   └── ...
│   │   ├── config.ts         # Frontend config
│   │   └── index.tsx         # React entry point
│   ├── package.json          # Node dependencies
│   └── build/                # Production build (generated)
├── docs/                      # Documentation
├── datasets/                  # Dataset storage (auto-created)
├── docker-compose.yml         # Docker services
├── Dockerfile                # Multi-stage build
├── .env.example              # Environment template
└── README.md                 # Project overview
```

## Technology Stack

### Backend
- **Framework**: FastAPI
- **Python**: 3.10+
- **Database**: SQLite (default), PostgreSQL (production)
- **MQTT**: Mosquitto broker
- **ML**: PyTorch, TensorFlow, ONNX Runtime, Ultralytics

### Frontend
- **Framework**: React 18
- **Language**: TypeScript
- **UI**: Custom component library
- **i18n**: react-i18next
- **Build**: Create React App

## Development Workflow

### 1. Start Development Environment

**Option A: Docker Compose (Recommended)**
```bash
docker compose up
```

**Option B: Local Development**
```bash
# Terminal 1: Backend
cd backend
python main.py

# Terminal 2: Frontend
cd frontend
npm start
```

### 2. Code Watch Mode

**Frontend** (hot reload on port 3000):
```bash
cd frontend
npm start
```

**Backend** (auto-reload on port 8000):
```bash
cd backend
uvicorn main:app --reload
```

### 3. Test Changes

1. Make code changes
2. Frontend auto-reloads, backend auto-restarts
3. Access http://localhost:3000 (frontend) or http://localhost:8000 (docker)
4. Verify changes work as expected

## Common Development Tasks

### Adding a New API Endpoint

**1. Backend** (`backend/api/routes.py`):
```python
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.post("/models/{model_id}/custom-action")
async def custom_action(
    model_id: int,
    param1: str,
    param2: int = 10
):
    """
    Description of what this endpoint does.

    Args:
        model_id: The model ID
        param1: Description of param1
        param2: Description of param2

    Returns:
        dict: Response data
    """
    try:
        # Your logic here
        result = do_something(model_id, param1, param2)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

**2. Frontend** (in component):
```typescript
const handleCustomAction = async (modelId: number) => {
  try {
    const formData = new FormData();
    formData.append('param1', 'value');
    formData.append('param2', '10');

    const response = await fetch(
      `${API_BASE_URL}/models/${modelId}/custom-action`,
      {
        method: 'POST',
        body: formData,
      }
    );

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      const errorMessage = typeof error.detail === 'string'
        ? error.detail
        : JSON.stringify(error.detail || error);
      throw new Error(errorMessage || t('key', 'Default error'));
    }

    const data = await response.json();
    showSuccess(t('key.successMessage', 'Success!'));
    // Handle success
  } catch (err: any) {
    console.error('Failed:', err);
    showError(err.message);
  }
};
```

**3. Add i18n translations**:
```json
// frontend/src/i18n/locales/en.json
{
  "modelSpace": {
    "customAction": "Custom Action",
    "customActionSuccess": "Action completed successfully",
    "customActionFailed": "Action failed"
  }
}
```

```json
// frontend/src/i18n/locales/zh.json
{
  "modelSpace": {
    "customAction": "自定义操作",
    "customActionSuccess": "操作成功完成",
    "customActionFailed": "操作失败"
  }
}
```

### Adding a New UI Component

**1. Create component** (`frontend/src/components/MyComponent.tsx`):
```typescript
import React from 'react';
import { useTranslation } from 'react-i18next';

interface MyComponentProps {
  value: string;
  onChange: (value: string) => void;
}

export const MyComponent: React.FC<MyComponentProps> = ({ value, onChange }) => {
  const { t } = useTranslation();

  return (
    <div className="my-component">
      <label>{t('myComponent.label', 'My Label')}</label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={t('myComponent.placeholder', 'Enter value')}
      />
    </div>
  );
};
```

**2. Use existing UI components**:
```typescript
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { FormField } from '../ui/FormField';

export const MyComponent: React.FC = () => {
  return (
    <FormField label="My Field" helpText="Help text">
      <Input value={value} onChange={setValue} />
      <Button onClick={handleSave}>Save</Button>
    </FormField>
  );
};
```

**3. Add translations** for all user-facing text.

### Error Handling Pattern

**Standard error handling (required for all API calls)**:
```typescript
if (!response.ok) {
  const error = await response.json().catch(() => ({}));
  const errorMessage = typeof error.detail === 'string'
    ? error.detail
    : JSON.stringify(error.detail || error);
  throw new Error(errorMessage || t('key.defaultError', 'Default message'));
}
```

**Why this pattern?**
- Handles both string and object error details
- Prevents "[object Object]" errors
- Provides clear error messages to users
- Supports i18n error messages

### i18n Pattern

**With variables**:
```typescript
t('key', 'Default {{var}} message', { var: value })
```

**Without variables**:
```typescript
t('key', 'Default message')
```

**In translation files**:
```json
{
  "key": "Translated {{var}} message"
}
```

**Why use this pattern?**
- Consistent across codebase
- Supports pluralization
- Type-safe with TypeScript
- Easy to add new languages

## Code Conventions

### TypeScript/React

1. **Functional components with hooks**:
   ```typescript
   export const MyComponent: React.FC<Props> = (props) => {
     // Hooks
     const [state, setState] = useState(initial);
     const { t } = useTranslation();

     // Effects
     useEffect(() => {
       // Effect logic
     }, [dependencies]);

     // Handlers
     const handleClick = () => {
       // Handle logic
     };

     // Render
     return <div>...</div>;
   };
   ```

2. **Use existing UI components**:
   - `Button`, `Input`, `Textarea`, `Select`
   - `FormField` for form layouts
   - `Alert` for notifications
   - `ConfirmDialog` for confirmations

3. **Always use i18n for user-facing text**:
   ```typescript
   // Good
   t('key', 'Default message')

   // Bad
   'Hardcoded text'
   ```

4. **Error handling**:
   - Always use standardized error pattern
   - Log errors with console.error
   - Show user-friendly messages

### Python/FastAPI

1. **Use type hints**:
   ```python
   def my_function(param1: str, param2: int) -> dict:
       ...
   ```

2. **Use async/await for I/O**:
   ```python
   async def my_endpoint():
       result = await some_async_operation()
       return result
   ```

3. **Return proper HTTP status codes**:
   ```python
   raise HTTPException(status_code=400, detail="Error message")
   ```

4. **Validate inputs**:
   ```python
   from pydantic import BaseModel, Field

   class MyModel(BaseModel):
       name: str = Field(..., min_length=1)
       count: int = Field(..., ge=0)
   ```

## Testing

### Manual Testing Checklist

Before committing changes, test:
- [ ] **Basic operations**: Upload, delete, edit resources
- [ ] **Error cases**: Invalid inputs, network errors
- [ ] **UI**: Check both English and Chinese
- [ ] **Console**: No errors in browser console (F12)
- [ ] **API**: Check backend logs for errors
- [ ] **Responsive**: Test on different screen sizes

### Common Test Scenarios

**Model Upload**:
- Valid .pt file
- Invalid file type (should show error)
- Missing required fields (should show validation)
- With calibration images (verify count display)

**Quantization**:
- Valid calibration ZIP
- Invalid ZIP (should show error)
- Check console for detailed errors

**Project Association**:
- Associate to project
- Dissociate from project
- Verify project names display correctly

## Debugging

### Frontend

**Browser Console** (F12):
```javascript
// Check for errors
console.error('Debug info:', data);

// Network tab to see API calls
// Response tab to see error details
```

**React DevTools**:
- Inspect component state and props
- Debug re-renders
- Profile performance

### Backend

**Check logs**:
```bash
docker compose logs -f camthink
```

**Debug endpoint**:
```python
@router.get("/debug")
async def debug_info():
    return {
        "env": settings.DEBUG,
        "working_dir": os.getcwd(),
        "python_version": sys.version
    }
```

### Common Issues

**"Module not found" (frontend)**:
```bash
cd frontend
npm install
```

**"Module not found" (backend)**:
```bash
cd backend
pip install -r requirements.txt
```

**Port already in use**:
```bash
# Find process
lsof -i :8000

# Kill process
kill -9 <PID>
```

**Docker container won't start**:
```bash
# Check logs
docker compose logs camthink

# Rebuild
docker compose build --no-cache camthink
```

## Recent Changes (2026-03-05)

### Error Handling Fixes
- Fixed "[object Object]" errors across 5 locations
- Standardized error message parsing
- Improved error logging

### Validation Improvements
- Added Num Classes validation (>= 1)
- Auto-sync Class Names with Num Classes
- Better user feedback

### i18n Improvements
- Fixed parameter passing in translations
- Consistent i18n pattern enforcement
- Calibration count display fix

See [CHANGELOG.md](../CHANGELOG.md) for full details.

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [React i18next](https://react.i18next.com/)
- [Docker Documentation](https://docs.docker.com/)
