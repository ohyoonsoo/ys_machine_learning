#include "ysException.hpp"

YSException::YSException(const char* message)
    : m_message { message } {}

const char* YSException::what() {
    return m_message;
}
