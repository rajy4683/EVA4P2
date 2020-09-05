#include <cmath>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe
{

namespace
{
constexpr char normRectTag[] = "NORM_RECT";
constexpr char normalizedLandmarkListTag[] = "NORM_LANDMARKS";
constexpr char recognizedHandGestureTag[] = "RECOGNIZED_HAND_GESTURE";
} // namespace

// Graph config:
//
// node {
//   calculator: "ActionTriggerCalculator"
//   input_stream: "NORM_LANDMARKS:scaled_landmarks"
//   input_stream: "NORM_RECT:hand_rect_for_next_frame"
// }
class ActionTriggerCalculator : public CalculatorBase
{
public:
    static ::mediapipe::Status GetContract(CalculatorContract *cc);
    ::mediapipe::Status Open(CalculatorContext *cc) override;

    ::mediapipe::Status Process(CalculatorContext *cc) override;

private:
    float get_Euclidean_DistanceAB(float a_x, float a_y, float b_x, float b_y)
    {
        float dist = std::pow(a_x - b_x, 2) + pow(a_y - b_y, 2);
        return std::sqrt(dist);
    }

    bool isThumbNearFirstFinger(NormalizedLandmark point1, NormalizedLandmark point2)
    {
        float distance = this->get_Euclidean_DistanceAB(point1.x(), point1.y(), point2.x(), point2.y());
        return distance < 0.1;
    }
};

REGISTER_CALCULATOR(ActionTriggerCalculator);

::mediapipe::Status ActionTriggerCalculator::GetContract(
    CalculatorContract *cc)
{
    /*RET_CHECK(cc->Inputs().HasTag(normalizedLandmarkListTag));
    cc->Inputs().Tag(normalizedLandmarkListTag).Set<mediapipe::NormalizedLandmarkList>();

    RET_CHECK(cc->Inputs().HasTag(normRectTag));
    cc->Inputs().Tag(normRectTag).Set<NormalizedRect>();
    */
    RET_CHECK(cc->Inputs().HasTag(recognizedHandGestureTag));
    cc->Inputs().Tag(recognizedHandGestureTag).Set<std::string>();
    //predicted_gesture = cc->InputSidePackets().Tag(recognizedHandGestureTag).Get<std::string>();

    return ::mediapipe::OkStatus();
}

::mediapipe::Status ActionTriggerCalculator::Open(
    CalculatorContext *cc)
{
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
}

::mediapipe::Status ActionTriggerCalculator::Process(
    CalculatorContext *cc)
{
    std::string *recognized_hand_gesture;

    // hand closed (red) rectangle
    const std::string& predicted_gesture = cc->Inputs().Tag(recognizedHandGestureTag).Get<std::string>();

    //RET_CHECK(splited_file_path.size() >= 2)
    RET_CHECK_GT(predicted_gesture.size(), 0) << "Predicted string is empty.";
    std::string mqtt_pub_cmd = absl::StrCat("/usr/bin/mosquitto_pub -h 127.0.0.1 -p 1883 -t 'Hellos/Pi3' -m '",predicted_gesture,"'");

    // finger states
    //LOG(INFO) << mqtt_pub_cmd ;
    system(mqtt_pub_cmd.c_str());

    /*cc->Outputs()
        .Tag(recognizedHandGestureTag)
        .Add(recognized_hand_gesture, cc->InputTimestamp());
    */
    return ::mediapipe::OkStatus();
} // namespace mediapipe

} // namespace mediapipe
