from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from llm.client import llm_client
from schemas.llm import ChatRequest, ChatResponse


router = APIRouter(prefix="/llm", tags=["LLM"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        answer = llm_client.chat(
            user_message=req.message,
            system_prompt=req.system_prompt,
        )
        return ChatResponse(answer=answer)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM request failed: {str(e)}"
        )


@router.post("/stream_chat")
def stream_chat(req: ChatRequest):
    try:
        if not req.message or not req.message.strip():
            raise ValueError("message cannot be empty")

        return StreamingResponse(
            llm_client.stream_chat(
                user_message=req.message,
                system_prompt=req.system_prompt,
            ),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM stream request failed: {str(e)}"
        )


@router.post("/json_chat", response_model=dict)
def json_chat(req: ChatRequest):
    try:
        result = llm_client.json_chat(
            user_message=req.message,
            system_prompt=req.system_prompt,
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM JSON request failed: {str(e)}"
        )
