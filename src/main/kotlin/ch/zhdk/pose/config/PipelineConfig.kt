package ch.zhdk.pose.config

import ch.bildspur.model.DataModel
import ch.bildspur.ui.properties.*
import ch.zhdk.pose.pipeline.PipelineType
import com.google.gson.annotations.Expose

class PipelineConfig {
    @Expose
    var calibration = CalibrationConfig()

    @Expose
    @EnumParameter("Pipeline")
    var pipelineType = DataModel(PipelineType.LightOpenPoseTorch)

    @BooleanParameter("Processing Enabled")
    var enabled = DataModel(true)

    @Expose
    @BooleanParameter("Annotate Output")
    var annotateOutput = DataModel(true)

    @Expose
    @BooleanParameter("Display Output")
    var displayOutput = DataModel(true)

    @LabelParameter("Information")
    val informationLabel = ""

    @StringParameter("Input Width", isEditable = false)
    var inputWidth = DataModel(1)

    @StringParameter("Input Height", isEditable = false)
    var inputHeight = DataModel(1)

    @StringParameter("Frame Time", isEditable = false)
    var frameTime = DataModel("- ms")

    @StringParameter("Processing Time", isEditable = false)
    var processingTime = DataModel("- ms")

    @StringParameter("FPS", isEditable = false)
    var framesPerSecond = DataModel("-")

    @LabelParameter("Pre-Processing")
    private val inputLabel = ""

    @Expose
    @BooleanParameter("Enable Pre-Processing")
    var enablePreProcessing = DataModel(true)

    @Expose
    @SliderParameter("Gamma Correction", 0.04, 4.0, 0.01)
    var gammaCorrection = DataModel(1.0)

    @LabelParameter("Detection")
    private val detectionLabel = ""

    @Expose
    @SliderParameter("Confidence Threshold", 0.0, 1.0, 0.05, snap = true)
    var threshold = DataModel(0.5)

    @Expose
    @SliderParameter("Image Scale", 0.05, 1.0, 0.05, snap = true)
    var imageScale = DataModel(1.0)
}