from logger import logger


if __name__ == "__main__":
    logger.info("Starting server...")
    import uvicorn
    uvicorn.run("price_predictor:app", host="0.0.0.0", port=8000, reload=True)
