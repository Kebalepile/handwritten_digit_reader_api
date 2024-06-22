from apis.predict_api_v_0 import  load_model_before_fork, app

if __name__ == '__main__':
    load_model_before_fork()
    app.run(host='0.0.0.0', port=10000)
