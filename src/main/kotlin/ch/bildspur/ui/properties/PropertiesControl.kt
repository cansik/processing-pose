package ch.bildspur.ui.properties

import ch.bildspur.event.Event
import ch.bildspur.ui.properties.types.*
import javafx.geometry.Insets
import javafx.geometry.Pos
import javafx.scene.control.Label
import javafx.scene.layout.HBox
import javafx.scene.layout.VBox
import javafx.scene.text.Font
import java.lang.reflect.Field

class PropertiesControl : VBox() {

    val propertyChanged = Event<BaseProperty>()

    init {
        spacing = 10.0
        alignment = Pos.TOP_CENTER
        padding = Insets(10.0, 20.0, 10.0, 10.0)
    }

    fun initView(obj: Any) {
        clearView()

        val params = readParameters(obj)

        // create view
        params.forEach {
            if (it.isAnnotationPresent(StringParameter::class.java)) {
                val annotation = it.getAnnotation(StringParameter::class.java)
                addProperty(annotation.name, StringProperty(it, obj, annotation))
            }

            if (it.isAnnotationPresent(NumberParameter::class.java)) {
                val annotation = it.getAnnotation(NumberParameter::class.java)
                addProperty(annotation.name, NumberProperty(it, obj, annotation))
            }

            if (it.isAnnotationPresent(SliderParameter::class.java)) {
                val annotation = it.getAnnotation(SliderParameter::class.java)
                addProperty(annotation.name, SliderProperty(it, obj, annotation))
            }

            if (it.isAnnotationPresent(BooleanParameter::class.java)) {
                val annotation = it.getAnnotation(BooleanParameter::class.java)
                addProperty(annotation.name, BooleanProperty(it, obj, annotation))
            }

            if (it.isAnnotationPresent(PVectorParameter::class.java)) {
                val annotation = it.getAnnotation(PVectorParameter::class.java)
                addProperty(annotation.name, PVectorProperty(it, obj, annotation))
            }

            if (it.isAnnotationPresent(ActionParameter::class.java)) {
                val annotation = it.getAnnotation(ActionParameter::class.java)
                addProperty(annotation.name, ActionProperty(it, obj, annotation))
            }

            if (it.isAnnotationPresent(EnumParameter::class.java)) {
                val annotation = it.getAnnotation(EnumParameter::class.java)
                addProperty(annotation.name, EnumProperty(it, obj, annotation))
            }

            if (it.isAnnotationPresent(PVectorAngleParameter::class.java)) {
                val annotation = it.getAnnotation(PVectorAngleParameter::class.java)
                addProperty(annotation.name, PVectorAngleProperty(it, obj, annotation))
            }

            if (it.isAnnotationPresent(RangeSliderParameter::class.java)) {
                val annotation = it.getAnnotation(RangeSliderParameter::class.java)
                addProperty(annotation.name, RangeSliderProperty(it, obj, annotation))
            }

            if (it.isAnnotationPresent(TextParameter::class.java)) {
                val annotation = it.getAnnotation(TextParameter::class.java)
                addProperty(annotation.name, TextProperty(it, obj, annotation))
            }

            if (it.isAnnotationPresent(Float2Parameter::class.java)) {
                val annotation = it.getAnnotation(Float2Parameter::class.java)
                addProperty(annotation.name, Float2Property(it, obj, annotation))
            }

            if (it.isAnnotationPresent(Float3Parameter::class.java)) {
                val annotation = it.getAnnotation(Float3Parameter::class.java)
                addProperty(annotation.name, Float3Property(it, obj, annotation))
            }

            if (it.isAnnotationPresent(LabelParameter::class.java)) {
                val annotation = it.getAnnotation(LabelParameter::class.java)
                addLabel(LabelProperty(it, obj, annotation))
            }
        }
    }

    fun clearView() {
        this.children.clear()
    }

    private fun addLabel(propertyView: BaseProperty) {
        children.add(propertyView)
    }

    private fun addProperty(name: String, propertyView: BaseProperty) {
        propertyView.propertyChanged += {
            propertyChanged(propertyView)
        }

        val nameLabel = Label("$name:")
        nameLabel.prefWidth = 100.0
        nameLabel.font = Font("Helvetica", 12.0)
        nameLabel.isWrapText = true

        val box = HBox(nameLabel, propertyView)
        box.spacing = 10.0
        box.prefHeight = propertyView.prefHeight
        box.alignment = Pos.CENTER_LEFT
        children.add(box)
    }

    private fun readParameters(obj: Any): List<Field> {
        val c = obj.javaClass

        val fields = c.declaredFields.filter {
            it.isAnnotationPresent(SliderParameter::class.java) ||
                    it.isAnnotationPresent(StringParameter::class.java) ||
                    it.isAnnotationPresent(BooleanParameter::class.java) ||
                    it.isAnnotationPresent(NumberParameter::class.java) ||
                    it.isAnnotationPresent(PVectorParameter::class.java) ||
                    it.isAnnotationPresent(ActionParameter::class.java) ||
                    it.isAnnotationPresent(EnumParameter::class.java) ||
                    it.isAnnotationPresent(PVectorAngleParameter::class.java) ||
                    it.isAnnotationPresent(LabelParameter::class.java) ||
                    it.isAnnotationPresent(TextParameter::class.java) ||
                    it.isAnnotationPresent(Float2Parameter::class.java) ||
                    it.isAnnotationPresent(Float3Parameter::class.java) ||
                    it.isAnnotationPresent(RangeSliderParameter::class.java)

        }
        fields.forEach { it.isAccessible = true }
        return fields
    }
}