from price_predictor import app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("price_predictor:app", host="0.0.0.0", port=8000, reload=True)
