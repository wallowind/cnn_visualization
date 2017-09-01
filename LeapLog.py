import logging


def get_log(module_name, log_level):
    '''
Function create log, based on module __name__ attribute, with custom log_level
    '''
    log = logging.getLogger(module_name)
    formatter = logging.Formatter(
        '%(levelname)s' + '[%(filename)s:%(funcName)s:%(lineno)d] %(message)s')
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    levels = {
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL}
    try:
        ch.setLevel(levels[log_level])
    except KeyError:
        print('There is no such level as {0}, use one of follow parameter:\
                \n{1}\n{2}\n{3}\n{4}\n{5}'.format(log_level, 'DEBUG',
                                                  'INFO', 'WARNING',
                                                  'ERROR', 'CRITICAL'))
        print('Level of logging set to default: WARNING')
        ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log
