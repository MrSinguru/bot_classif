from model import ClassPredictor
from telegram_token import token
import torch
from config import reply_texts
import numpy as np
from PIL import Image
from io import BytesIO
from multiprocessing import Queue, Process
from time import sleep


model = ClassPredictor()
job_queue = Queue()


def worker(bot, queue):
    while True:
        message = queue.get()
        chat_id = message.chat_id
        print("Got image from {}".format(chat_id))

        # получаем информацию о картинке
        image_info = message.photo[-1]
        image_file = bot.get_file(image_info)
        image_stream = BytesIO()
        image_file.download(out=image_stream)

        # симулируем долгую работу сети, чтобы показать, почему нужен отдельный процесс и как это работает
        class_ = 1#model.predict(image_stream)
        sleep(10)

        # теперь отправим результат
        message.reply_text(str(class_))
        print("Sent Answer to user, predicted: {}".format(class_))


def photo(bot, update):
    update.message.reply_text("Ваше фото помещено в очередь")
    job_queue.put(update.message)


if __name__ == '__main__':
    from telegram.ext import Updater, MessageHandler, Filters
    import logging

    # Включим самый базовый логгинг, чтобы видеть сообщения об ошибках
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    # используем прокси, так как без него у меня ничего не работало(
    updater = Updater(token=token)
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, photo))

    # Сделаем отдельный процесс для того, чтобы обрабатывать картинки
    worker_args = (updater.bot, job_queue)
    worker_process = Process(target=worker, args=worker_args)
    worker_process.start()

    updater.start_polling()
