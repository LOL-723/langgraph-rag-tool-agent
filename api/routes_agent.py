from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from llm.client import llm_client


router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post("/ask", response_model=dict)
def Agent_Ask(
    message: str = Form(...),
    system_prompt: str | None = Form(default=None),
    use_rag: bool = Form(default=False),
    file: UploadFile | None = File(default=None),
):
    try:
        return llm_client.Agent_Ask(
            user_message=message,
            system_prompt=system_prompt,
            file=file,
            use_rag=use_rag,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent request failed: {str(e)}"
        )
