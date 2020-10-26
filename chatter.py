from optparse import OptionParser
from model.train import train
from model.response import response
import common.data_utils as data_utils
from config.get_config import get_config


class CmdParser(OptionParser):
    def error(self, msg):
        print("ERROR!提示信息如下：")
        self.print_help()
        self.exit(0)

    def exit(self, status=0, msg=None):
        exit(status)


if __name__ == '__main__':
    parser = CmdParser()
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="请执行正确的指令模式： -t pre_treat/train/chat")
    (option, args) = parser.parse_args()
    config = get_config()

    if option.type == "pre_treat":
        data_utils.preprocess_raw_data(raw_data=config["corpus"], tokenized_data=config["tokenized_corpus"])
    elif option.type == "train":
        train()
    elif option.type == "chat":
        print("Agent: 你好！结束聊天请输入ESC。")
        while True:
            req = input("User：")
            if req == "ESC":
                print("Agent：再见！")
                exit(0)
            res = response(sentence=req)
            print("Agent：", res)
    else:
        parser.error(msg='')

